"""Phase 59 — real-LLM-driven multi-round delayed-disambiguation.

SDK v3.13, W12 family anchor. **First cross-round real-LLM transfer
benchmark** in the CoordPy programme.

This phase is the *real-LLM transfer* slice of the SDK v3.12 W11
synthetic win on R-58. Where Phase-58 hand-built canonical
``claim_kind`` strings and canonical ``service=<tag>`` payload tokens
on every emitted handoff, Phase-59 puts a *real-LLM-shaped extractor*
between the operational events and the team-coord candidate stream:

* **Round 1** — operational events (latency spikes, error rates,
  firewall denials) on (gold_A, gold_B, decoy) are presented to the
  ``monitor`` and ``network`` producer roles via an LLM-style prompt;
  each role's "response" is parsed by the same closed-vocabulary
  parser (``parse_role_response`` from Phase 53). Synthesised noise:
  the LLM may emit a synonym for a known kind (``DEADLOCK_DETECTED``
  vs ``DEADLOCK_SUSPECTED``), drop the ``service=`` token in favour
  of an alternative spelling (``svc=X`` / ``for service X``), or omit
  a claim entirely.
* **Round 2** — diagnostic events on db_admin / sysadmin contain a
  *specific* causal clue (deadlock / pool exhaustion / disk fill /
  slow query). Same LLM prompt, same parser, same noise.

The candidate stream the auditor sees is therefore *closed-vocabulary
in the post-normalisation form* but *drift-prone in the pre-
normalisation form*. The two methods we compare:

* The SDK v3.12 :class:`MultiRoundBundleDecoder` (W11) consumes the
  raw stream — it cannot match drifted ``claim_kind``s against its
  ``_DECODER_PRIORITY`` table, so the elected root_cause stays
  generic and the W11 contradiction-aware drop never fires. This is
  **W12-Λ** at the real-LLM axis.
* The new SDK v3.13 :class:`RobustMultiRoundBundleDecoder` (W12)
  applies the closed-vocabulary normalisation layer ahead of W11.
  Under the named bounded-noise budget, the post-normalisation stream
  is shape-equivalent to the R-58 ground-truth stream; W11-1
  sufficiency carries over (W12-1).

LLM modes
---------

Phase-59 supports three extractor modes, all running through the same
post-extractor pipeline:

* ``synthetic_noisy_llm`` (default, CI-runnable) — a deterministic
  in-process extractor that mimics real-LLM kind drift + payload drift
  using two named knobs (``synonym_prob``, ``svc_token_alt_prob``).
  This is the **W12 anchor mode**: deterministic, reproducible from
  ``bank_seed``, no network. Calibrated against the Phase-53 14B / 35B
  parser_role_response empirical kind-drift histograms.
* ``synthetic_clean_llm`` — the same deterministic extractor with the
  noise knobs at zero. Backward-compat anchor: must reduce to R-58.
* ``ollama`` (opt-in) — real Ollama backend over the role prompts
  defined in Phase-53. Requires a live endpoint; falls back to the
  noisy synthetic extractor when the endpoint is unreachable. Used
  for W12-C2 conjecture probing.

Theorem cross-reference (W12 family)
------------------------------------

* **W12-Λ** (proved-empirical) — single-round / un-normalised W11
  collapse on R-59 default.
* **W12-1** (proved-conditional, empirical) — RobustMultiRound
  sufficiency under bounded LLM noise.
* **W12-2** (proved by inspection) — the closed-vocabulary
  normalisation table is sound: every entry maps to a canonical kind
  in ``_DECODER_PRIORITY``.
* **W12-3** (proved-empirical) — backward-compat with R-58:
  RobustMultiRoundBundleDecoder ties MultiRoundBundleDecoder on
  Phase-58 default and on R-59 ``synthetic_clean_llm`` mode.
* **W12-4** (proved-empirical) — noise-budget falsifier: when
  ``synonym_prob`` exceeds the closed-vocabulary table coverage and
  the LLM emits *unknown* kinds, normalisation cannot recover and W12
  ties FIFO at 0.000.

CLI
---

::

    # Default (synthetic_noisy_llm, CI-runnable, W12-1 anchor):
    python3 -m vision_mvp.experiments.phase59_real_llm_multi_round \\
        --K-auditor 8 --n-eval 12 --out -

    # Clean baseline (W12-3 backward-compat anchor):
    python3 -m vision_mvp.experiments.phase59_real_llm_multi_round \\
        --llm-mode synthetic_clean_llm --K-auditor 8 --n-eval 12 --out -

    # Falsifier bank (W12-4 anchor):
    python3 -m vision_mvp.experiments.phase59_real_llm_multi_round \\
        --falsifier --K-auditor 8 --n-eval 12 --out -

    # Cross-regime (R-54..R-58 default + R-59 default):
    python3 -m vision_mvp.experiments.phase59_real_llm_multi_round \\
        --cross-regime --n-eval 8 --out -

    # 5-seed stability sweep:
    python3 -m vision_mvp.experiments.phase59_real_llm_multi_round \\
        --seed-sweep --n-eval 12 --out -

    # Real Ollama (opt-in; requires live endpoint):
    python3 -m vision_mvp.experiments.phase59_real_llm_multi_round \\
        --llm-mode ollama --endpoint http://192.168.12.191:11434 \\
        --model qwen2.5:14b-32k --n-eval 8 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
import time
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    IncidentScenario,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.coordpy.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.coordpy.llm_backend import LLMBackend, OllamaBackend
from vision_mvp.coordpy.team_coord import (
    AdmissionPolicy, BundleAwareTeamDecoder,
    CLAIM_KIND_SYNONYMS,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    RobustMultiRoundBundleDecoder, RoleBudget,
    TeamCoordinator, audit_team_lifecycle,
    collect_admitted_handoffs, _DecodedHandoff,
    normalize_claim_kind, normalize_payload,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    MultiRoundScenario, build_phase58_bank, build_phase58_falsifier_bank,
    _build_round_candidates, _as_incident_scenario,
)


# =============================================================================
# Synthetic-noisy-LLM extractor — drift-prone but deterministic
# =============================================================================
#
# Calibrated against the empirical Phase-53 14B / 35B parser_role_response
# behaviour: we observed that real-LLM emissions drift from canonical
# kinds to plausible synonyms ~15-25% of the time on the closed-vocab
# incident-triage prompt, and drop the ``service=`` token in favour of
# alternative phrasings ~10% of the time. The synthetic noisy extractor
# parameterises both rates so the W12 contract tests can falsify the
# claim ``"normalisation is load-bearing"`` mechanically.


# Per-canonical-kind alternative spellings the LLM might emit. Lex-
# ordered for diff stability. Every entry must be a key of
# CLAIM_KIND_SYNONYMS, and CLAIM_KIND_SYNONYMS[entry] must be the
# canonical key. This is what makes W12-2 mechanically checkable.
NOISY_KIND_VARIANTS: dict[str, tuple[str, ...]] = {
    "DEADLOCK_SUSPECTED": (
        "DEADLOCK_DETECTED", "DEADLOCK", "DEADLOCK_OBSERVED",
        "LOCK_CYCLE",
    ),
    "POOL_EXHAUSTION": (
        "POOL_EXHAUSTED", "CONNECTION_POOL_EXHAUSTED",
        "DB_POOL_FULL", "POOL_FULL", "POOL_SATURATED",
    ),
    "SLOW_QUERY_OBSERVED": (
        "SLOW_QUERY", "SLOW_QUERIES", "QUERY_SLOWDOWN",
    ),
    "DISK_FILL_CRITICAL": (
        "DISK_FILL", "DISK_FULL", "DISK_USAGE_CRITICAL",
        "DISK_NEAR_FULL",
    ),
    "ERROR_RATE_SPIKE": (
        "ERROR_RATE", "ERROR_SPIKE", "ERROR_SURGE", "HIGH_ERROR_RATE",
    ),
    "LATENCY_SPIKE": (
        "LATENCY", "HIGH_LATENCY", "P95_HIGH", "SLO_BREACH",
    ),
    "FW_BLOCK_SURGE": (
        "FIREWALL_BLOCKS", "FW_DENY", "BLOCKED_PACKETS",
    ),
    "TLS_EXPIRED": ("CERT_EXPIRED", "TLS_EXPIRY"),
    "DNS_MISROUTE": ("DNS_FAILURE", "DNS_SERVFAIL"),
    "OOM_KILL": ("OOM", "OOM_KILLED", "OUT_OF_MEMORY"),
    "CRON_OVERRUN": ("CRON_TIMEOUT", "CRON_LATE"),
}


# Alternative service-tag spellings the LLM might emit. Mirrors the
# normaliser's ``_SERVICE_TAG_REWRITES`` patterns so contract tests
# can verify W12 closure across both directions.
SERVICE_TAG_ALT_TEMPLATES: tuple[str, ...] = (
    "svc={tag}",
    "service:{tag}",
    "for service {tag}",
    "on service {tag}",
    "service_name={tag}",
    "svc_name={tag}",
)


# Out-of-vocabulary kinds the LLM might emit for the W12-4 falsifier
# bank. None of these appear in CLAIM_KIND_SYNONYMS — normalisation
# cannot rescue them. Two roles' worth so the falsifier bank can
# saturate every causal kind.
OUT_OF_VOCAB_KINDS: dict[str, str] = {
    "DEADLOCK_SUSPECTED": "DEADLOCK_PROBABLY_DETECTED_MAYBE",
    "POOL_EXHAUSTION": "POOL_LOOKING_BUSY",
    "SLOW_QUERY_OBSERVED": "QUERY_SOMEWHAT_SLUGGISH",
    "DISK_FILL_CRITICAL": "DISK_GETTING_FULL_PROBABLY",
}


@dataclasses.dataclass
class NoisyLLMExtractorConfig:
    """Per-bench knobs for the synthetic noisy-LLM extractor.

    Calibration target (from Phase-53 empirical extractor stats):

    * ``synonym_prob = 0.50`` — half of all causal claims emerge in
      a synonym variant. This is *above* the empirical 14B / 35B
      drift rate (we observed 15-25%) but well *below* the
      out-of-vocab regime (W12-4). The default is intentionally
      aggressive: a normaliser that can only rescue 25% noise is
      a weak normaliser; we want to demonstrate the W12 method
      under realistic-to-pessimistic drift.
    * ``svc_token_alt_prob = 0.30`` — payload-level token drift.
    * ``oov_prob = 0.0`` — out-of-vocab drift (only set in the
      falsifier bank).
    * ``drop_claim_prob = 0.0`` — drop the claim entirely
      (W12-aux research; not in the default).
    """
    synonym_prob: float = 0.50
    svc_token_alt_prob: float = 0.30
    oov_prob: float = 0.0
    drop_claim_prob: float = 0.0
    seed: int = 11

    def is_clean(self) -> bool:
        return (self.synonym_prob == 0.0
                and self.svc_token_alt_prob == 0.0
                and self.oov_prob == 0.0
                and self.drop_claim_prob == 0.0)


def _maybe_drift_kind(
        canonical_kind: str,
        cfg: NoisyLLMExtractorConfig,
        rng: random.Random,
        ) -> str | None:
    """Return the LLM's emitted kind for a causal canonical kind.

    Returns ``None`` to mean "the LLM dropped this claim entirely".
    The drift order is (drop, oov, synonym, identity).
    """
    if cfg.drop_claim_prob > 0 and rng.random() < cfg.drop_claim_prob:
        return None
    if cfg.oov_prob > 0 and rng.random() < cfg.oov_prob:
        oov = OUT_OF_VOCAB_KINDS.get(canonical_kind)
        if oov is not None:
            return oov
    if cfg.synonym_prob > 0 and rng.random() < cfg.synonym_prob:
        variants = NOISY_KIND_VARIANTS.get(canonical_kind, ())
        if variants:
            return variants[rng.randrange(0, len(variants))]
    return canonical_kind


def _maybe_drift_payload(
        payload: str,
        cfg: NoisyLLMExtractorConfig,
        rng: random.Random,
        ) -> str:
    """Apply payload-level token drift: rewrite ``service=<tag>``
    into one of the alternative spellings with probability
    ``svc_token_alt_prob``."""
    if cfg.svc_token_alt_prob <= 0 or not payload:
        return payload
    out = payload
    # Rewrite each service= occurrence with the configured probability.
    def _rewrite(m: re.Match) -> str:
        if rng.random() >= cfg.svc_token_alt_prob:
            return m.group(0)
        tag = m.group(1)
        template = SERVICE_TAG_ALT_TEMPLATES[
            rng.randrange(0, len(SERVICE_TAG_ALT_TEMPLATES))]
        return template.format(tag=tag)
    return re.sub(r"\bservice=([\w-]+)", _rewrite, out)


@dataclasses.dataclass
class NoisyLLMExtractor:
    """Drift-prone deterministic extractor.

    Consumes a Phase-58-style ``MultiRoundScenario`` and produces the
    candidate handoff stream that an LLM-with-bounded-drift would have
    emitted. The drift is deterministic given ``cfg.seed`` and the
    scenario_id (each scenario has its own RNG so inter-scenario
    interference is impossible).
    """

    cfg: NoisyLLMExtractorConfig

    def extract_round(self,
                       scenario: MultiRoundScenario,
                       round_idx: int,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        emissions = (scenario.round1_emissions if round_idx == 1
                       else scenario.round2_emissions)
        # Per-(scenario, round) RNG for full determinism. Use a
        # SHA-256-based deterministic seed (Python's built-in
        # ``hash()`` is salted across process restarts unless
        # ``PYTHONHASHSEED`` is set, which would make the bench's
        # noise channel irreproducible).
        import hashlib as _hashlib
        seed_key = (f"{self.cfg.seed}|{scenario.scenario_id}"
                     f"|round{round_idx}").encode("utf-8")
        seed_int = int.from_bytes(
            _hashlib.sha256(seed_key).digest()[:8], "big")
        rng = random.Random(seed_int)
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            for (canonical_kind, canonical_payload) in emissions.get(
                    role, ()):
                drifted_kind = _maybe_drift_kind(
                    canonical_kind, self.cfg, rng)
                if drifted_kind is None:
                    continue
                drifted_payload = _maybe_drift_payload(
                    canonical_payload, self.cfg, rng)
                # The substrate subscription table is keyed on the
                # *canonical* kind — drifted kinds may not match. We
                # route under the canonical (so the bench is fair),
                # but the handoff carries the drifted kind/payload so
                # the un-normalised pipeline sees real drift.
                consumers = subs.consumers(role, canonical_kind)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, drifted_kind,
                                  drifted_payload, (0,)))
        return out


# =============================================================================
# Real Ollama extractor — opt-in
# =============================================================================
#
# When --llm-mode ollama is set, the same Phase-53-style prompt is sent
# to the configured LLM and the response is parsed by the closed-
# vocabulary parser. The extractor degrades gracefully on HTTP failure.


def _round_ollama_prompt(
        role: str,
        round_idx: int,
        emissions: Sequence[tuple[str, str]],
        allowed_kinds: Sequence[str],
        ) -> str:
    """Render the Phase-53-style role prompt for Phase-59. Same closed
    vocabulary; we add a one-line note about the round so the LLM
    knows whether it is in symptom-mode (round 1) or diagnostic-mode
    (round 2)."""
    kind_lines = "\n".join(f"  - {k}" for k in allowed_kinds)
    event_lines = []
    for i, (canon_kind, payload) in enumerate(emissions, start=1):
        # Drop the canonical kind from the event description; the LLM
        # must infer the kind from the body.
        event_lines.append(
            f"  [{i}] body=\"{payload}\"")
    if not event_lines:
        event_lines = ["  (none)"]
    round_hint = ("operational symptoms (latency/error/firewall)"
                   if round_idx == 1
                   else "specific diagnostic clues (deadlock/pool/disk/query)")
    return (
        f"You are the {role!r} agent in an incident-response team. "
        f"This is round {round_idx}: {round_hint}.\n\n"
        f"Allowed claim kinds for {role!r}:\n{kind_lines}\n\n"
        f"Events you observed:\n"
        + "\n".join(event_lines) + "\n\n"
        f"For each event, emit ONE LINE in the format\n"
        f"  KIND | one-line evidence including any service token\n"
        f"Output rules: only KINDs from the list. One claim per line. "
        f"Maximum 6 lines. If no claim, output NONE.\n\n"
        f"Begin output now:\n"
    )


def _parse_ollama_response(
        response: str,
        allowed: Sequence[str],
        max_claims: int = 6,
        ) -> list[tuple[str, str]]:
    """Closed-vocabulary parser. Same shape as
    :func:`vision_mvp.experiments.phase53_scale_vs_structure.parse_role_response`
    but admissive of synonym variants (we *want* the LLM to drift; the
    Phase-59 driver tests whether normalisation can rescue the drift).
    """
    if not response:
        return []
    line_re = re.compile(r"^\s*([A-Z][A-Z0-9_]*)\s*[|:\-–—]\s*(.+?)\s*$")
    allowed_set = set(allowed)
    syn_set = set(CLAIM_KIND_SYNONYMS.keys())
    out: list[tuple[str, str]] = []
    seen = set()
    for raw in response.splitlines():
        line = raw.strip()
        if not line or line.upper() == "NONE":
            continue
        m = line_re.match(line)
        if not m:
            continue
        kind = m.group(1).upper()
        payload = m.group(2).strip()
        # Accept either the canonical allowed kind or any synonym we
        # know about — that is how we let the LLM drift.
        if kind not in allowed_set and kind not in syn_set:
            continue
        if kind in seen:
            continue
        seen.add(kind)
        out.append((kind, payload[:240]))
        if len(out) >= max_claims:
            break
    return out


@dataclasses.dataclass
class OllamaLLMExtractor:
    """Opt-in real-LLM extractor via :class:`OllamaBackend`.

    Falls back to the noisy synthetic extractor when the endpoint is
    unreachable (HTTP error). The fallback is **labelled** in the
    returned report so the cross-regime table is honest about which
    scenarios used real-LLM extraction vs synthetic fallback.
    """

    backend: LLMBackend
    fallback_cfg: NoisyLLMExtractorConfig
    n_real_calls: int = 0
    n_failed_calls: int = 0
    total_wall_s: float = 0.0
    n_synthetic_fallbacks: int = 0

    def extract_round(self,
                       scenario: MultiRoundScenario,
                       round_idx: int,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        emissions = (scenario.round1_emissions if round_idx == 1
                       else scenario.round2_emissions)
        from vision_mvp.core.extractor_noise import (
            incident_triage_known_kinds)
        known = incident_triage_known_kinds()
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            role_emissions = emissions.get(role, ())
            if not role_emissions:
                continue
            allowed = tuple(known.get(role, ()))
            if not allowed:
                continue
            prompt = _round_ollama_prompt(role, round_idx,
                                            role_emissions, allowed)
            t0 = time.time()
            try:
                resp = self.backend.generate(
                    prompt, max_tokens=160, temperature=0.0)
                self.n_real_calls += 1
                self.total_wall_s += time.time() - t0
            except Exception:
                self.n_failed_calls += 1
                self.total_wall_s += time.time() - t0
                # Deterministic synthetic fallback for this role/round.
                fallback = NoisyLLMExtractor(self.fallback_cfg)
                role_out = [
                    e for e in fallback.extract_round(scenario, round_idx)
                    if e[0] == role]
                self.n_synthetic_fallbacks += 1
                out.extend(role_out)
                continue
            parsed = _parse_ollama_response(resp, allowed)
            for (kind, payload) in parsed:
                # Find consumers under either the canonical kind (if
                # the LLM emitted canonical) or the canonical that the
                # synonym resolves to.
                canon = CLAIM_KIND_SYNONYMS.get(kind, kind)
                consumers = subs.consumers(role, canon)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
        return out


# =============================================================================
# Phase 59 driver
# =============================================================================


def _run_capsule_strategy(
        scenario: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        round2_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        ) -> StrategyResult:
    """Run two coordination rounds against the *real* (or noisy-LLM-
    extracted) candidate stream. Mirrors phase58's run-driver but
    accepts pre-extracted per-round candidates."""
    incident_sc = _as_incident_scenario(scenario)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="real_llm_phase59",
    )
    coord.advance_round(1)
    for (src, to, kind, payload, _evs) in round1_cands:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv1 = coord.role_view_cid(ROLE_AUDITOR)

    coord.advance_round(1)
    coord.policy_per_role = policy_per_role_factory(round_idx=2,
                                                       cands=round2_cands)
    for (src, to, kind, payload, _evs) in round2_cands:
        coord.emit_handoff(
            source_role=src, to_role=to, claim_kind=kind, payload=payload)
    coord.seal_all_role_views()
    rv2 = coord.role_view_cid(ROLE_AUDITOR)

    if decoder_mode == "robust_multi_round":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        decoder = RobustMultiRoundBundleDecoder()
        answer = decoder.decode_rounds([union])
    elif decoder_mode == "multi_round_bundle":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        decoder = MultiRoundBundleDecoder()
        answer = decoder.decode_rounds([union])
    elif decoder_mode == "single_round_bundle":
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])
        decoder = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=2)
        answer = decoder.decode(r2_handoffs)
    else:  # per_round substrate-style
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])

        @dataclasses.dataclass(frozen=True)
        class _Shim:
            source_role: str
            claim_kind: str
            payload: str
            n_tokens: int = 1

        shimmed = [_Shim(h.source_role, h.claim_kind, h.payload)
                    for h in r2_handoffs]
        answer = _phase31_decoder_from_handoffs(shimmed)

    coord.seal_team_decision(
        team_role=ROLE_AUDITOR, decision=answer,
        extra_role_view_cids=[rv1] if rv1 and rv1 != rv2 else ())
    audit = audit_team_lifecycle(ledger)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))

    rv2_cap = ledger.get(rv2) if rv2 else None
    n_admitted_r2 = (rv2_cap.payload.get("n_admitted")
                       if rv2_cap is not None
                       and isinstance(rv2_cap.payload, dict) else 0)
    n_tokens_r2 = (rv2_cap.payload.get("n_tokens_admitted", 0)
                     if rv2_cap is not None
                     and isinstance(rv2_cap.payload, dict) else 0)
    admitted_kinds: set[tuple[str, str]] = set()
    for cid in (rv1, rv2):
        if not cid or cid not in ledger:
            continue
        cap = ledger.get(cid)
        for p in cap.parents:
            if p in ledger:
                handoff = ledger.get(p)
                if handoff.kind != CapsuleKind.TEAM_HANDOFF:
                    continue
                payload = (handoff.payload
                            if isinstance(handoff.payload, dict) else {})
                admitted_kinds.add((str(payload.get("source_role", "")),
                                     str(payload.get("claim_kind", ""))))
    required = {(role, kind)
                 for (role, kind, _p, _evs) in incident_sc.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    return StrategyResult(
        strategy=strategy_name,
        scenario_id=scenario.scenario_id,
        answer=answer,
        grading=grading,
        failure_kind=failure_kind,
        n_admitted_auditor=int(n_admitted_r2 or 0),
        n_dropped_auditor_budget=0,
        n_dropped_auditor_capacity=0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=coord.stats()["n_team_handoff"],
        n_role_view=coord.stats()["n_role_view"],
        n_team_decision=coord.stats()["n_team_decision"],
        audit_ok=audit.is_ok(),
        n_tokens_admitted=int(n_tokens_r2 or 0),
    )


def _run_substrate_strategy(
        scenario: MultiRoundScenario,
        round1_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        round2_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        inbox_capacity: int,
        ) -> StrategyResult:
    """Substrate baseline: route round-1 + round-2 candidates through
    the substrate inbox-router; decode round-2 holdings only (same
    shape as phase58's substrate baseline)."""
    incident_sc = _as_incident_scenario(scenario)
    from vision_mvp.core.role_handoff import HandoffRouter, RoleInbox
    subs = build_role_subscriptions()
    router = HandoffRouter(subs=subs)
    for role in ALL_ROLES:
        router.register_inbox(RoleInbox(role=role, capacity=inbox_capacity))
    for round_idx, cands in ((1, round1_cands), (2, round2_cands)):
        for (src, _to, kind, payload, evids) in cands:
            router.emit(
                source_role=src,
                source_agent_id=ALL_ROLES.index(src),
                claim_kind=kind, payload=payload,
                source_event_ids=evids, round=round_idx)
    auditor_inbox = router.inboxes.get(ROLE_AUDITOR)
    held = tuple(auditor_inbox.peek()) if auditor_inbox else ()
    answer = _phase31_decoder_from_handoffs(held)
    grading = grade_answer(incident_sc, _format_canonical_answer(answer))
    admitted_kinds = {(h.source_role, h.claim_kind) for h in held}
    required = {(role, kind)
                 for (role, kind, _p, _evs) in incident_sc.causal_chain}
    if grading["full_correct"]:
        failure_kind = "none"
    elif required - admitted_kinds:
        failure_kind = "missing_handoff"
    else:
        failure_kind = "decoder_error"
    return StrategyResult(
        strategy="substrate",
        scenario_id=scenario.scenario_id,
        answer=answer,
        grading=grading,
        failure_kind=failure_kind,
        n_admitted_auditor=len(held),
        n_dropped_auditor_budget=auditor_inbox.n_overflow if auditor_inbox else 0,
        n_dropped_auditor_capacity=auditor_inbox.n_dedup if auditor_inbox else 0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=0, n_role_view=0, n_team_decision=0,
        audit_ok=False,
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


# =============================================================================
# Bench-property metrics — verifying the LLM-drifted bench property
# =============================================================================


_GENERIC_NOISE_AFTER_NORMALIZATION = frozenset({
    "LATENCY_SPIKE", "ERROR_RATE_SPIKE", "FW_BLOCK_SURGE",
})


def _bench_property(
        scenario: MultiRoundScenario,
        round1_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        round2_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        ) -> dict[str, Any]:
    """Pre-committed mechanical witnesses of the Phase-59 bench
    property — the *same shape* as Phase-58's bench property, but
    measured *after normalisation* (so that synonym drift is folded
    into the canonical kinds for the property check).
    """
    def _norm_kinds(cands):
        return {normalize_claim_kind(c[2]) for c in cands
                  if c[1] == ROLE_AUDITOR}
    def _has_service(cands, decoy):
        for (_s, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            np = normalize_payload(payload)
            if f"service={decoy}" in np:
                return True
        return False
    def _decoy_roles(cands, decoy):
        roles = set()
        for (src, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            np = normalize_payload(payload)
            if f"service={decoy}" in np:
                roles.add(src)
        return roles
    r1_kinds_norm = _norm_kinds(round1_cands)
    r2_kinds_norm = _norm_kinds(round2_cands)
    decoy = scenario.decoy_storm_service
    round1_only_generic = bool(r1_kinds_norm) and r1_kinds_norm.issubset(
        _GENERIC_NOISE_AFTER_NORMALIZATION)
    round2_only_specific = bool(r2_kinds_norm) and not (
        r2_kinds_norm & _GENERIC_NOISE_AFTER_NORMALIZATION)
    r2_mentions_decoy = _has_service(round2_cands, decoy)
    r1_decoy_roles = _decoy_roles(round1_cands, decoy)
    n_drifted_r1 = sum(
        1 for c in round1_cands if c[1] == ROLE_AUDITOR
        and c[2] != normalize_claim_kind(c[2]))
    n_drifted_r2 = sum(
        1 for c in round2_cands if c[1] == ROLE_AUDITOR
        and c[2] != normalize_claim_kind(c[2]))
    return {
        "round1_only_generic_noise": round1_only_generic,
        "round2_only_specific": round2_only_specific,
        "decoy_only_in_round1": (not r2_mentions_decoy),
        "round1_decoy_corroborated": len(r1_decoy_roles) >= 2,
        "delayed_causal_evidence_property_holds": (
            round1_only_generic and round2_only_specific
            and (not r2_mentions_decoy)
            and len(r1_decoy_roles) >= 2),
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
        "n_kind_drifted_round1": n_drifted_r1,
        "n_kind_drifted_round2": n_drifted_r2,
    }


# =============================================================================
# Top-level driver
# =============================================================================


def _make_factory(name: str, priorities, budgets):
    """Return a ``policy_per_role_factory(round_idx, cands) -> dict``
    for the given strategy. Mirrors phase58's factory."""
    def fac(round_idx: int = 1,
             cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]] | None = None,
             ) -> dict[str, AdmissionPolicy]:
        if name == "capsule_fifo":
            return {r: FifoAdmissionPolicy() for r in budgets}
        if name == "capsule_priority":
            return {r: ClaimPriorityAdmissionPolicy(
                priorities=priorities, threshold=0.65) for r in budgets}
        if name == "capsule_coverage":
            return {r: CoverageGuidedAdmissionPolicy() for r in budgets}
        cands = cands or []
        cands_aud = [c for c in cands if c[1] == ROLE_AUDITOR]
        if name == "capsule_cohort_buffered":
            policy = (CohortCoherenceAdmissionPolicy
                      .from_candidate_payloads([c[3] for c in cands_aud]))
            return {r: policy for r in budgets}
        if name == "capsule_corroboration":
            policy = (CrossRoleCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud]))
            return {r: policy for r in budgets}
        if name in ("capsule_multi_service",
                     "capsule_bundle_decoder"):
            policy = (MultiServiceCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud],
                          top_k=3, min_corroborated_roles=2))
            return {r: policy for r in budgets}
        if name in ("capsule_multi_round", "capsule_robust_multi_round"):
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
    return fac


def _build_extractor(llm_mode: str,
                       cfg: NoisyLLMExtractorConfig,
                       endpoint: str,
                       model: str,
                       timeout_s: float,
                       ):
    """Construct an extractor object from the named LLM mode.

    Returns one of:
      * :class:`NoisyLLMExtractor` (synthetic_clean_llm with cfg=clean,
        synthetic_noisy_llm with cfg as given)
      * :class:`OllamaLLMExtractor` (real backend with synthetic
        fallback on HTTP failure)
    """
    if llm_mode == "synthetic_clean_llm":
        return NoisyLLMExtractor(NoisyLLMExtractorConfig(
            synonym_prob=0.0, svc_token_alt_prob=0.0,
            oov_prob=0.0, drop_claim_prob=0.0, seed=cfg.seed))
    if llm_mode == "synthetic_noisy_llm":
        return NoisyLLMExtractor(cfg)
    if llm_mode == "ollama":
        backend = OllamaBackend(
            model=model, base_url=endpoint, timeout=timeout_s,
            think=(False if "qwen3" in model else None))
        return OllamaLLMExtractor(
            backend=backend, fallback_cfg=cfg)
    raise ValueError(
        f"unknown llm_mode {llm_mode!r}; "
        f"valid: 'synthetic_clean_llm', 'synthetic_noisy_llm', 'ollama'")


def run_phase59(*,
                  n_eval: int | None = None,
                  K_auditor: int = 8,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 3,
                  use_falsifier_bank: bool = False,
                  llm_mode: str = "synthetic_noisy_llm",
                  llm_synonym_prob: float = 0.50,
                  llm_svc_alt_prob: float = 0.30,
                  llm_oov_prob: float = 0.0,
                  llm_seed: int = 11,
                  endpoint: str = "http://192.168.12.191:11434",
                  model: str = "qwen2.5:14b-32k",
                  timeout_s: float = 300.0,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 59 end-to-end.

    Default config (W12-1 anchor): ``llm_mode='synthetic_noisy_llm'``,
    ``synonym_prob=0.50``, ``svc_token_alt_prob=0.30``,
    ``K_auditor=8``, ``n_eval=12`` (4 builders × 3 replicates),
    ``bank_seed=11``.
    """
    if use_falsifier_bank:
        bank = build_phase58_falsifier_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    else:
        bank = build_phase58_bank(
            n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    cfg = NoisyLLMExtractorConfig(
        synonym_prob=llm_synonym_prob,
        svc_token_alt_prob=llm_svc_alt_prob,
        oov_prob=(0.50 if use_falsifier_bank else llm_oov_prob),
        drop_claim_prob=0.0,
        seed=llm_seed)
    extractor = _build_extractor(
        llm_mode=llm_mode, cfg=cfg, endpoint=endpoint,
        model=model, timeout_s=timeout_s)

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    strategies: list[tuple[str, str]] = [
        ("capsule_fifo", "per_round"),
        ("capsule_priority", "per_round"),
        ("capsule_coverage", "per_round"),
        ("capsule_cohort_buffered", "per_round"),
        ("capsule_corroboration", "per_round"),
        ("capsule_multi_service", "per_round"),
        ("capsule_bundle_decoder", "single_round_bundle"),
        ("capsule_multi_round", "multi_round_bundle"),
        ("capsule_robust_multi_round", "robust_multi_round"),
    ]

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    for sc in bank:
        # Extract the LLM-shaped per-round candidate streams ONCE per
        # scenario; every strategy then re-runs against the same
        # streams for an apples-to-apples comparison.
        round1_cands = extractor.extract_round(sc, 1)
        round2_cands = extractor.extract_round(sc, 2)
        bench_property_per_scenario[sc.scenario_id] = _bench_property(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in strategies:
            fac = _make_factory(sname, priorities, budgets)
            results.append(_run_capsule_strategy(
                scenario=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands))

    strategy_names = ("substrate",) + tuple(s[0] for s in strategies)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "robust_multi_round_minus_fifo": gap(
            "capsule_robust_multi_round", "capsule_fifo"),
        "robust_multi_round_minus_substrate": gap(
            "capsule_robust_multi_round", "substrate"),
        "robust_multi_round_minus_multi_round": gap(
            "capsule_robust_multi_round", "capsule_multi_round"),
        "robust_multi_round_minus_bundle_decoder": gap(
            "capsule_robust_multi_round", "capsule_bundle_decoder"),
        "max_non_robust_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_robust_multi_round"),
    }

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    bench_summary = {
        "scenarios_with_property": sum(
            1 for v in bench_property_per_scenario.values()
            if v["delayed_causal_evidence_property_holds"]),
        "n_scenarios": len(bench_property_per_scenario),
        "K_auditor": K_auditor,
        "n_kind_drifted_round1_total": sum(
            v["n_kind_drifted_round1"]
            for v in bench_property_per_scenario.values()),
        "n_kind_drifted_round2_total": sum(
            v["n_kind_drifted_round2"]
            for v in bench_property_per_scenario.values()),
    }

    extractor_stats: dict[str, Any] = {"llm_mode": llm_mode,
                                          "noise_cfg": dataclasses.asdict(cfg)}
    if isinstance(extractor, OllamaLLMExtractor):
        extractor_stats["n_real_calls"] = extractor.n_real_calls
        extractor_stats["n_failed_calls"] = extractor.n_failed_calls
        extractor_stats["total_wall_s"] = round(
            extractor.total_wall_s, 3)
        extractor_stats["n_synthetic_fallbacks"] = (
            extractor.n_synthetic_fallbacks)

    if verbose:
        print(f"[phase59] llm_mode={llm_mode}, n_eval={len(bank)}, "
              f"K_auditor={K_auditor}, falsifier={use_falsifier_bank}",
              file=sys.stderr, flush=True)
        print(f"[phase59] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank)}",
              file=sys.stderr, flush=True)
        print(f"[phase59] kind-drifted handoffs (r1/r2): "
              f"{bench_summary['n_kind_drifted_round1_total']}/"
              f"{bench_summary['n_kind_drifted_round2_total']}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase59]   {s:32s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase59] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)

    return {
        "schema": "phase59.real_llm_multi_round.v1",
        "config": {
            "n_eval": len(bank), "K_auditor": K_auditor,
            "T_auditor": T_auditor, "K_producer": K_producer,
            "T_producer": T_producer, "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "use_falsifier_bank": use_falsifier_bank,
            "llm_mode": llm_mode,
            "llm_synonym_prob": llm_synonym_prob,
            "llm_svc_alt_prob": llm_svc_alt_prob,
            "llm_oov_prob": (0.50 if use_falsifier_bank else llm_oov_prob),
            "llm_seed": llm_seed,
            "endpoint": endpoint,
            "model": model,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "extractor_stats": extractor_stats,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }


def run_phase59_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 12, K_auditor: int = 8, T_auditor: int = 256,
        llm_mode: str = "synthetic_noisy_llm",
        llm_synonym_prob: float = 0.50,
        llm_svc_alt_prob: float = 0.30,
        ) -> dict[str, Any]:
    """5-seed stability sweep over both ``bank_seed`` and ``llm_seed``
    set to the same value (so each seed picks a distinct combined
    bank-and-noise configuration). The W12-1 strong-bar anchor."""
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase59(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=3,
            llm_mode=llm_mode,
            llm_synonym_prob=llm_synonym_prob,
            llm_svc_alt_prob=llm_svc_alt_prob,
            llm_seed=seed, verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
        }
    return {
        "schema": "phase59.real_llm_multi_round_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor, "T_auditor": T_auditor,
        "n_eval": n_eval,
        "llm_mode": llm_mode,
        "llm_synonym_prob": llm_synonym_prob,
        "llm_svc_alt_prob": llm_svc_alt_prob,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 8, bank_seed: int = 11,
                                ) -> dict[str, Any]:
    """Single regime table: R-54..R-58 default + R-59 default. The W12
    backward-compat audit anchor."""
    from vision_mvp.experiments.phase54_cross_role_coherence import (
        run_phase54)
    from vision_mvp.experiments.phase55_decoy_plurality import (
        run_phase55)
    from vision_mvp.experiments.phase56_multi_service_corroboration import (
        run_phase56)
    from vision_mvp.experiments.phase57_decoder_forcing import (
        run_phase57)
    from vision_mvp.experiments.phase58_multi_round_decoder import (
        run_phase58)
    p54 = run_phase54(n_eval=n_eval, K_auditor=4, T_auditor=128,
                       bank_seed=bank_seed,
                       bank_replicates=2, verbose=False)
    p55 = run_phase55(n_eval=n_eval, K_auditor=4, T_auditor=128,
                       bank_seed=bank_seed,
                       bank_replicates=2, use_falsifier_bank=False,
                       verbose=False)
    p56 = run_phase56(n_eval=n_eval, K_auditor=4, T_auditor=128,
                       bank_seed=bank_seed,
                       bank_replicates=2, use_falsifier_bank=False,
                       verbose=False)
    p57 = run_phase57(n_eval=n_eval, K_auditor=8, T_auditor=256,
                       bank_seed=bank_seed, bank_replicates=3,
                       use_falsifier_bank=False, verbose=False)
    p58 = run_phase58(n_eval=n_eval, K_auditor=8, T_auditor=256,
                       bank_seed=bank_seed, bank_replicates=2,
                       verbose=False)
    p59_clean = run_phase59(n_eval=n_eval, K_auditor=8, T_auditor=256,
                              bank_seed=bank_seed, bank_replicates=2,
                              llm_mode="synthetic_clean_llm",
                              verbose=False)
    p59_noisy = run_phase59(n_eval=n_eval, K_auditor=8, T_auditor=256,
                              bank_seed=bank_seed, bank_replicates=2,
                              llm_mode="synthetic_noisy_llm",
                              verbose=False)
    p59_falsifier = run_phase59(
        n_eval=n_eval, K_auditor=8, T_auditor=256,
        bank_seed=bank_seed, bank_replicates=2,
        use_falsifier_bank=True,
        llm_mode="synthetic_noisy_llm", verbose=False)
    return {
        "schema": "phase59.cross_regime.v1",
        "config": {"n_eval": n_eval, "bank_seed": bank_seed},
        "phase54_default": p54,
        "phase55_default": p55,
        "phase56_default": p56,
        "phase57_default": p57,
        "phase58_default": p58,
        "phase59_clean": p59_clean,
        "phase59_noisy": p59_noisy,
        "phase59_falsifier": p59_falsifier,
    }


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 59 — real-LLM-driven multi-round delayed-"
                    "disambiguation benchmark (SDK v3.13 / W12 family).")
    p.add_argument("--K-auditor", type=int, default=8)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=12)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=3)
    p.add_argument("--falsifier", action="store_true")
    p.add_argument("--cross-regime", action="store_true")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--llm-mode", type=str,
                    default="synthetic_noisy_llm",
                    choices=("synthetic_clean_llm",
                              "synthetic_noisy_llm", "ollama"))
    p.add_argument("--llm-synonym-prob", type=float, default=0.50)
    p.add_argument("--llm-svc-alt-prob", type=float, default=0.30)
    p.add_argument("--llm-oov-prob", type=float, default=0.0)
    p.add_argument("--llm-seed", type=int, default=11)
    p.add_argument("--endpoint", type=str,
                    default="http://192.168.12.191:11434")
    p.add_argument("--model", type=str, default="qwen2.5:14b-32k")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.cross_regime:
        report = run_cross_regime_summary(
            n_eval=args.n_eval, bank_seed=args.bank_seed)
    elif args.seed_sweep:
        report = run_phase59_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            llm_mode=args.llm_mode,
            llm_synonym_prob=args.llm_synonym_prob,
            llm_svc_alt_prob=args.llm_svc_alt_prob)
    else:
        report = run_phase59(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            use_falsifier_bank=args.falsifier,
            llm_mode=args.llm_mode,
            llm_synonym_prob=args.llm_synonym_prob,
            llm_svc_alt_prob=args.llm_svc_alt_prob,
            llm_oov_prob=args.llm_oov_prob,
            llm_seed=args.llm_seed,
            endpoint=args.endpoint, model=args.model,
            timeout_s=args.timeout, verbose=not args.quiet)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out == "-":
        print(text)
    elif args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        if not args.quiet:
            print(f"[phase59] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
