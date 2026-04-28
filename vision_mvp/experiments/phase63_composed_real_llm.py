"""Phase 63 — composed end-to-end W14 + W15 benchmark.

SDK v3.17, W16 family anchor. **First end-to-end composition move** in
the Wevra programme — W14 (producer-side ambiguity preservation) and
W15 (decoder-side capsule context packing) on a *single* regime where
both layers are independently load-bearing.

Recap of why composition needs its own bench
=============================================

SDK v3.15 (W14) closed the producer-side gap on R-61 against a real
Ollama 14B (``qwen2.5:14b-32k``) producer at +0.500 strict gain over
substrate FIFO under the structured prompt. SDK v3.16 (W15) closed the
decoder-side gap on R-62-tightbudget against the *deterministic
identity* extractor at +1.000 strict gain over FIFO-packed-W13.

But the two layers had not been stress-tested *together* on a regime
where:

* the producer compresses ambiguity upstream (so W14 alone gates whether
  the bench property survives), AND
* the decoder is bounded by a strict ``T_decoder`` token budget (so
  W15 alone gates whether the round-2 disambiguator survives FIFO
  truncation).

Phase-63 attacks that exact composition. The events are R-62's
multi-hypothesis comparable-magnitude shape (2 gold + 2 decoy services,
asymmetric corroboration: every decoy ≥ 2 distinct producer roles in
round 1, every gold = 1 role). The producer is Phase-61's
deterministic ``MagnitudeFilteringExtractor`` with its
``prompt_mode`` knob — naive prompt collapses round-1 toward top-N
events per (role, kind), structured prompt preserves the per-event
mandate. The decoder consumes the cross-round bundle under a strict
``T_decoder`` (default tight = ``24`` tokens, mirroring R-62-tightbudget).

The composition makes a *new* test possible that neither prior layer
covers alone: under naive prompt + tight T_decoder, **two distinct
structural limits** (W14-Λ-prompt + W15-Λ-budget) compose
multiplicatively and *no* capsule strategy in the SDK can clear 0.000.
Under structured prompt + tight T_decoder, the producer-side property
is restored (W14 closes its gap) AND the decoder-side budget pressure
remains (W15 closes its gap). Only the AttentionAwareBundleDecoder run
*on the structured-prompt stream* simultaneously closes both gaps —
the **W16-1 anchor**.

Pre-committed sub-banks
========================

Six sub-banks instantiate the W16 cross-regime. The default config in
each is ``K_auditor=12, T_auditor=256, n_eval=8, bank_seed=11`` (the
Phase-62 defaults) and the producer parameters are
``threshold_p95_ms=1000.0, threshold_error_rate=0.10,
threshold_fw_count=5`` (the Phase-61 calibration anchors) — every
parameter is pre-committed before any run.

* **R-63-baseline-loose** (identity producer, ``T_decoder=None``).
  Sanity anchor: every cross-round capsule decoder hits 1.000;
  W15-3 backward-compat anchor on R-62 events.
* **R-63-W15-only** (identity producer, ``T_decoder=24``). Identical
  to R-62-tightbudget: isolates the W15 contribution. W15 = 1.000,
  FIFO-pack = 0.000.
* **R-63-W14-only** (mag-filter naive, ``T_decoder=None``). Isolates
  the W14-Λ-prompt limit on R-62 events: under producer compression
  the bench property fails and every cross-round decoder collapses
  to 0.000.
* **R-63-naive-tight** (mag-filter naive, ``T_decoder=24``). Joint
  failure regime: W14-Λ-prompt × W15-Λ-budget compose; every
  capsule strategy ties FIFO at 0.000. The structural argument is
  the *additive* composition of the two named limits.
* **R-63-W14-success** (mag-filter structured, ``T_decoder=None``).
  W14 alone restores the bench property; cross-round decoders return
  to 1.000.
* **R-63-COMPOSED-TIGHT** (mag-filter structured, ``T_decoder=24``).
  **The W16-1 anchor.** W14 restores the bench property AND the
  decoder budget bites simultaneously. Only the AttentionAware
  decoder over the structured-prompt stream wins at 1.000;
  FIFO-packed-W13 (the W14-only-budgeted baseline) collapses to
  0.000; the AttentionAware decoder over the *naive-prompt* stream
  (the W15-only-without-W14 baseline) collapses to 0.000.

Plus a named falsifier:

* **R-63-degen-budget** (mag-filter structured, ``T_decoder=2``).
  W16-Λ-degenerate falsifier — the budget is so tight that even
  the round-2 specific claim does not fit (1 token of header
  margin). Both packers collapse; the win is conditional on a
  budget that admits *some* but not *all* of the union.

Plus an opt-in real-LLM probe via replay:

* **R-63-OLLAMA-REPLAY-LOOSE** (replay extractor, ``T_decoder=None``).
  Reads byte-for-byte the saved Phase-61 ``ollama_structured`` raw
  responses (``docs/data/phase61_real_ollama_structured_qwen2_5_14b
  _n8.json``) and runs the cross-round capsule pipeline. Expected
  outcome: matches the Phase-61 +0.500 anchor (W14-Λ-real).
* **R-63-OLLAMA-REPLAY-COMPOSED-TIGHT** (replay extractor,
  ``T_decoder=tight``). Same replay stream + W15 packer. The
  composed real-LLM measurement on the *recorded* qwen2.5:14b-32k
  bytes — honest to the empirical envelope of the Phase-61 probe.

The Ollama endpoint is **not** a hard dependency: the milestone is
fully evaluated on the deterministic synthetic counterparts; the
replay path is a *measurement* anchor over recorded real-LLM bytes,
not a fresh real-LLM probe.

Theorem cross-reference (W16 family)
=====================================

* **W16-Λ-compose** (proved-empirical + structural sketch on
  R-63-naive-tight). Under the magnitude-filter naive producer AND
  ``T_decoder = 24``, every capsule strategy in the SDK ties FIFO at
  ``accuracy_full = 0.000``. The bench property is erased upstream
  (W14-Λ-prompt) AND the decoder-side budget would drop the
  disambiguator anyway (W15-Λ-budget). The two limits compose
  multiplicatively on the *same* regime; closing one alone is
  insufficient.
* **W16-1** (proved-conditional + proved-empirical on
  R-63-COMPOSED-TIGHT). Pairing the W14
  :class:`StructuredProducerProtocol` (driving the magnitude-filter
  producer simulation) with the W15
  :class:`AttentionAwareBundleDecoder` (running the W13 layered
  decoder on a salience-packed subset under ``T_decoder``) achieves
  ``accuracy_full = 1.000`` while every non-composed baseline
  (W11/W12/W13 over a naive-prompt stream, W13 + FIFO pack over a
  structured-prompt stream, AttentionAware over a naive-prompt
  stream) ties FIFO at 0.000. Strict gain ≥ +1.000 vs every
  non-composed baseline; stable across 5/5 alternate ``bank_seed``
  values.
* **W16-2** (proved-empirical, sub-additivity). The composed gain
  exceeds the sum of W14-only and W15-only individual gains on
  R-63-COMPOSED-TIGHT: each layer alone is insufficient (one fails
  upstream, the other fails downstream); the composition is sharper
  than additive.
* **W16-3** (proved-empirical full programme regression). On
  R-63-baseline-loose and R-63-W14-success, the composition surface
  reduces to the prior milestones byte-for-byte: W15 with
  ``T_decoder=None`` ties W13 byte-for-byte (W15-3 anchor); the
  structured-prompt magnitude-filter ties the identity extractor's
  bench property (W14-3 anchor on R-62-shape events).
* **W16-Λ-real-replay** (empirical-research). On the recorded
  Phase-61 qwen2.5:14b-32k bytes (``ollama_structured`` mode,
  n=8, 24 producer calls, 0 endpoint failures, 111.4 s wall),
  applying the W15 packer over the cross-round capsule pipeline
  produces the *measured* real-LLM composed result — honest to the
  empirical envelope of the prior probe.
* **W16-C-real** (conjectural). A *fresh* live Ollama probe under
  R-63-COMPOSED-TIGHT with the structured prompt + tight T_decoder
  closes the 1/8 model-error failure that W14-only leaves — i.e.
  achieves accuracy_full ≥ 0.625 (= 5/8). Falsifier: the
  attention-pack does not change correctness on a real LLM whose
  attention is sufficiently uniform that prompt-position ordering
  does not matter. Not yet wired (Ollama endpoint offline at the
  time of milestone capture).

CLI
---

::

    # R-63-baseline-loose (sanity):
    python3 -m vision_mvp.experiments.phase63_composed_real_llm \\
        --extractor identity --prompt-mode naive \\
        --K-auditor 12 --n-eval 8 --out -

    # R-63-COMPOSED-TIGHT (W16-1 anchor):
    python3 -m vision_mvp.experiments.phase63_composed_real_llm \\
        --extractor magnitude_filter --prompt-mode structured \\
        --decoder-budget 24 --K-auditor 12 --n-eval 8 --out -

    # R-63 cross-regime (six synthetic sub-banks + falsifier):
    python3 -m vision_mvp.experiments.phase63_composed_real_llm \\
        --cross-regime --K-auditor 12 --n-eval 8 --out -

    # R-63 5-seed stability sweep on the composed regime:
    python3 -m vision_mvp.experiments.phase63_composed_real_llm \\
        --seed-sweep --extractor magnitude_filter \\
        --prompt-mode structured --decoder-budget 24 --n-eval 8 --out -

    # R-63-OLLAMA-REPLAY (uses recorded Phase-61 bytes):
    python3 -m vision_mvp.experiments.phase63_composed_real_llm \\
        --extractor ollama_replay \\
        --replay-source docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json \\
        --decoder-budget 24 --n-eval 8 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Any, Sequence

from vision_mvp.tasks.incident_triage import (
    ALL_ROLES, ROLE_AUDITOR, ROLE_DB_ADMIN, ROLE_MONITOR,
    ROLE_NETWORK, ROLE_SYSADMIN,
    build_role_subscriptions, grade_answer,
    _decoder_from_handoffs as _phase31_decoder_from_handoffs,
)
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, AttentionAwareBundleDecoder,
    BundleAwareTeamDecoder,
    CapsuleContextPacker,
    CLAIM_KIND_SYNONYMS,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy, FifoContextPacker,
    INCIDENT_TRIAGE_OBSERVATION_KINDS,
    LayeredClaimNormalizer,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
    ProducerPromptResult,
    RobustMultiRoundBundleDecoder, RoleBudget,
    StructuredProducerProtocol,
    TeamCoordinator, audit_team_lifecycle,
    collect_admitted_handoffs, _DecodedHandoff,
    incident_triage_role_schemas,
    normalize_payload,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    MultiRoundScenario, _as_incident_scenario,
)
from vision_mvp.experiments.phase61_producer_ambiguity_preservation import (
    MagnitudeFilteringExtractor,
    IdentityExtractor as Phase61IdentityExtractor,
    _parse_structured_response,
    _bench_property as _phase61_bench_property,
)
from vision_mvp.experiments.phase59_real_llm_multi_round import (
    _parse_ollama_response,
)
from vision_mvp.experiments.phase62_attention_aware_packing import (
    IdentityExtractor as Phase62IdentityExtractor,
    build_phase62_bank,
    _bench_property as _phase62_bench_property,
)


# =============================================================================
# Replay extractor — re-runs the cross-round capsule pipeline against
# the recorded raw responses from a Phase-61 real-Ollama probe.
# =============================================================================
#
# The replay extractor reads ``docs/data/phase61_real_ollama_structured_
# qwen2_5_14b_n8.json`` (or any compatible Phase-61 ollama capture) and
# parses each saved raw response through the same parser the live driver
# uses. The parsed claims are routed via the same role subscriptions, so
# the resulting candidate stream is byte-for-byte equivalent to the live
# driver's output on the same scenario.
#
# This is *honest replay* — we do not synthesise responses; we replay
# the model bytes recorded in the prior milestone and apply the W15
# packer to them. The outcome is bounded to the prior milestone's
# empirical envelope.
#
# IMPORTANT scenario-id mapping. Phase-61 scenarios use the
# ``compmag`` family (single-decoy R-61 events); Phase-62/63 scenarios
# use the ``multi`` family (multi-hypothesis R-62 events). The replay
# extractor matches by *event-shape and role pattern*, not by raw
# scenario_id, by mapping each Phase-61 raw response onto the
# corresponding Phase-61 scenario (so the replay regime is R-61 events
# under W14, plus an optional W15 layer at decode time). When the
# Phase-63 driver is invoked with ``extractor=ollama_replay``, the
# scenario bank switches to the Phase-61 bank to keep replay coherent.


@dataclasses.dataclass
class OllamaReplayExtractor:
    """Replay-only extractor that reads recorded raw responses from a
    Phase-61 real-Ollama report and re-routes them through the same
    parser + role-subscription pipeline. No live LLM calls.

    Attributes
    ----------
    raw_responses_per_scenario
        ``{scenario_id: [{role, round_idx, text, prompt_mode, ...}, ...]}``
        as captured by ``CapturingOllamaExtractor`` in
        :mod:`vision_mvp.experiments.phase61_producer_ambiguity_preservation`.
    prompt_mode
        Recorded prompt mode (``"structured"`` for the W14 anchor,
        ``"naive"`` for the W14-Λ-real anchor). The parser dispatches
        accordingly.
    n_replay_calls / n_missing_keys
        Forensic counters: how many (scenario, round, role) triples were
        successfully replayed, and how many fell through because no
        recorded response covered that triple (silently emits no claims).
    """

    raw_responses_per_scenario: dict[str, list[dict[str, Any]]]
    prompt_mode: str = PRODUCER_PROMPT_STRUCTURED
    source_path: str = ""
    n_replay_calls: int = 0
    n_missing_keys: int = 0

    def reset_counters(self) -> None:
        self.n_replay_calls = 0
        self.n_missing_keys = 0

    @classmethod
    def from_phase61_report(cls, path: str) -> "OllamaReplayExtractor":
        """Load a Phase-61 real-Ollama report from disk and return a
        replay extractor over its raw responses.
        """
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        raw = d.get("raw_responses_per_scenario") or {}
        if not raw:
            raise ValueError(
                f"replay source {path!r} has no raw_responses_per_scenario "
                f"block (was the report captured with --extractor ollama?)")
        prompt_mode = (d.get("config", {}).get("prompt_mode")
                        or PRODUCER_PROMPT_STRUCTURED)
        return cls(
            raw_responses_per_scenario={
                k: list(v) for k, v in raw.items()
            },
            prompt_mode=prompt_mode,
            source_path=path,
        )

    def replay_scenarios(self) -> list[str]:
        """The list of scenario_ids covered by this replay source."""
        return sorted(self.raw_responses_per_scenario.keys())

    def extract_round(self,
                       sc: MultiRoundScenario,
                       round_idx: int,
                       *,
                       protocol: StructuredProducerProtocol | None = None,
                       schemas: dict[str, "RoleExtractionSchema"] | None = None,
                       record_prompts: list[ProducerPromptResult] | None = None,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        del protocol, schemas, record_prompts  # unused on replay
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        entries = self.raw_responses_per_scenario.get(sc.scenario_id, [])
        for entry in entries:
            if int(entry.get("round_idx", 0)) != int(round_idx):
                continue
            role = str(entry.get("role", ""))
            if role not in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                              ROLE_NETWORK):
                continue
            text = str(entry.get("text", ""))
            mode = str(entry.get("prompt_mode", self.prompt_mode))
            # Build the kinds-in-scope set the same way the live driver
            # does (the Phase-61 schema for this role).
            schemas_local = incident_triage_role_schemas()
            schema = schemas_local.get(role)
            if schema is None:
                continue
            allowed = (schema.diagnosis_kinds if int(round_idx) == 2
                        else schema.observation_kinds)
            allowed_full = sorted(set(allowed) | set(schema.allowed_kinds))
            if mode == PRODUCER_PROMPT_STRUCTURED:
                parsed = _parse_structured_response(text, allowed_full)
            else:
                parsed = _parse_ollama_response(text, allowed_full)
            self.n_replay_calls += 1
            for (kind, payload) in parsed:
                canon = CLAIM_KIND_SYNONYMS.get(kind, kind)
                consumers = subs.consumers(role, canon)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
        if not entries:
            self.n_missing_keys += 1
        return out


# =============================================================================
# Bench-property dispatcher — Phase-62 multi-hypothesis vs Phase-61 single-decoy
# =============================================================================


def _bench_property_for(sc: MultiRoundScenario,
                          round1_cands,
                          round2_cands) -> dict[str, Any]:
    """Return the bench-property witness block appropriate for the scenario
    family. Phase-63 uses Phase-62 multi-hypothesis events as the default
    bank; the optional ollama_replay path uses Phase-61 single-decoy
    scenarios (so the bench-property witness must be Phase-61-shaped).
    """
    sid = sc.scenario_id
    if sid.startswith("p62_"):
        return _phase62_bench_property(sc, round1_cands, round2_cands)
    return _phase61_bench_property(sc, round1_cands, round2_cands)


# =============================================================================
# Strategy / decoder dispatch (same pattern as Phase-62)
# =============================================================================


def _make_factory(name: str, priorities, budgets):
    def fac(round_idx: int = 1, cands=None,
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
        if name in ("capsule_multi_service",):
            policy = (MultiServiceCorroborationAdmissionPolicy
                      .from_candidate_stream(
                          [(c[0], c[3]) for c in cands_aud],
                          top_k=4, min_corroborated_roles=2))
            return {r: policy for r in budgets}
        if name in ("capsule_multi_round",
                     "capsule_robust_multi_round",
                     "capsule_layered_multi_round",
                     "capsule_layered_fifo_packed",
                     "capsule_attention_aware"):
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
    return fac


def _decode_with_packer(union, packer, T_decoder, round_index_hint):
    """Decode via FIFO or salience packer; identical to Phase-62 helper.
    Kept inline to avoid cross-module coupling on a private helper."""
    first = MultiRoundBundleDecoder().decode_rounds([union])
    elected = str(first.get("root_cause", "unknown"))
    pack = packer.pack(union, elected_root_cause=elected,
                          T_decoder=T_decoder,
                          round_index_hint=round_index_hint)
    kept = [k.handoff for k in pack.kept]
    ans = LayeredRobustMultiRoundBundleDecoder().decode_rounds([kept])
    stats = pack.as_dict()
    stats["n_tokens_decoder_input"] = int(pack.tokens_kept)
    stats["n_handoffs_decoder_input"] = int(pack.n_kept)
    stats["n_handoffs_admitted"] = int(pack.n_input)
    return ans, stats


def _round_hint_from_ledger(ledger, role_view_cids: Sequence[str]
                              ) -> list[int]:
    """Derive a per-handoff round index by looking up each handoff's
    parent role-view's ``round`` payload field. Returns a list aligned
    with :func:`collect_admitted_handoffs`'s deduplicated union."""
    seen: set[str] = set()
    hint: list[int] = []
    for cid in role_view_cids:
        if not cid or cid not in ledger:
            continue
        cap = ledger.get(cid)
        r_idx = (cap.payload.get("round", 0)
                   if isinstance(cap.payload, dict) else 0)
        for p in cap.parents:
            if p in seen or p not in ledger:
                continue
            if ledger.get(p).kind != CapsuleKind.TEAM_HANDOFF:
                continue
            seen.add(p)
            hint.append(int(r_idx) or 1)
    return hint


def _run_capsule_strategy(
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands, round2_cands,
        T_decoder: int | None = None,
        ) -> tuple[StrategyResult, dict[str, Any]]:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase63_composed_real_llm",
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

    pack_stats: dict[str, Any] = {}

    if decoder_mode == "attention_aware":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_packer(
            union, CapsuleContextPacker(),
            T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "fifo_packed_layered":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        round_hint = _round_hint_from_ledger(ledger, (rv1, rv2))
        answer, pack_stats = _decode_with_packer(
            union, FifoContextPacker(),
            T_decoder=T_decoder,
            round_index_hint=round_hint)
    elif decoder_mode == "layered_multi_round":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        decoder = LayeredRobustMultiRoundBundleDecoder()
        answer = decoder.decode_rounds([union])
    elif decoder_mode == "robust_multi_round":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        answer = RobustMultiRoundBundleDecoder().decode_rounds([union])
    elif decoder_mode == "multi_round_bundle":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        answer = MultiRoundBundleDecoder().decode_rounds([union])
    elif decoder_mode == "single_round_bundle":
        r2_handoffs = collect_admitted_handoffs(ledger, [rv2])
        decoder = BundleAwareTeamDecoder(
            cck_filter=True, role_corroboration_floor=1,
            fallback_admitted_size_threshold=2)
        answer = decoder.decode(r2_handoffs)
    else:
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
    result = StrategyResult(
        strategy=strategy_name,
        scenario_id=sc.scenario_id,
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
    return result, pack_stats


def _run_substrate_strategy(
        sc: MultiRoundScenario,
        round1_cands, round2_cands, inbox_capacity) -> StrategyResult:
    incident_sc = _as_incident_scenario(sc)
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
        strategy="substrate", scenario_id=sc.scenario_id,
        answer=answer, grading=grading, failure_kind=failure_kind,
        n_admitted_auditor=len(held),
        n_dropped_auditor_budget=auditor_inbox.n_overflow if auditor_inbox else 0,
        n_dropped_auditor_capacity=auditor_inbox.n_dedup if auditor_inbox else 0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=0, n_role_view=0, n_team_decision=0,
        audit_ok=False,
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


# =============================================================================
# Phase 63 driver
# =============================================================================


_R63_STRATEGIES: tuple[tuple[str, str], ...] = (
    ("capsule_fifo", "per_round"),
    ("capsule_priority", "per_round"),
    ("capsule_coverage", "per_round"),
    ("capsule_cohort_buffered", "per_round"),
    ("capsule_corroboration", "per_round"),
    ("capsule_multi_service", "per_round"),
    ("capsule_multi_round", "multi_round_bundle"),
    ("capsule_robust_multi_round", "robust_multi_round"),
    ("capsule_layered_multi_round", "layered_multi_round"),
    # W14-only-budgeted baseline.
    ("capsule_layered_fifo_packed", "fifo_packed_layered"),
    # W14+W15 composed (when wired with the structured prompt
    # producer); also serves as the W15-only-on-naive cell when the
    # producer is mag-filter naive.
    ("capsule_attention_aware", "attention_aware"),
)


def _build_extractor(extractor_kind: str, *,
                      prompt_mode: str,
                      seed: int,
                      threshold_p95_ms: float,
                      threshold_error_rate: float,
                      threshold_fw_count: int,
                      replay_source: str = ""):
    """Construct the extractor object for Phase-63."""
    if extractor_kind == "identity":
        return Phase62IdentityExtractor(seed=seed)
    if extractor_kind == "magnitude_filter":
        return MagnitudeFilteringExtractor(
            seed=seed,
            threshold_p95_ms=threshold_p95_ms,
            threshold_error_rate=threshold_error_rate,
            threshold_fw_count=threshold_fw_count,
            prompt_mode=prompt_mode)
    if extractor_kind == "ollama_replay":
        if not replay_source:
            raise ValueError(
                "extractor=ollama_replay requires --replay-source <path>")
        return OllamaReplayExtractor.from_phase61_report(replay_source)
    raise ValueError(
        f"unknown extractor {extractor_kind!r}; valid: "
        f"identity / magnitude_filter / ollama_replay")


def _build_bank_for(extractor_kind: str, *,
                      n_replicates: int, bank_seed: int
                      ) -> list[MultiRoundScenario]:
    """The Phase-63 default bank is Phase-62 multi-hypothesis. The
    ollama_replay path requires the Phase-61 single-decoy bank because
    the recorded raw responses are keyed on Phase-61 scenario_ids.
    """
    if extractor_kind == "ollama_replay":
        from vision_mvp.experiments.phase61_producer_ambiguity_preservation import (
            build_phase61_bank,
        )
        return build_phase61_bank(n_replicates=n_replicates, seed=bank_seed)
    return build_phase62_bank(n_replicates=n_replicates, seed=bank_seed)


def run_phase63(*,
                 n_eval: int | None = None,
                 K_auditor: int = 12,
                 T_auditor: int = 256,
                 K_producer: int = 6,
                 T_producer: int = 96,
                 inbox_capacity: int | None = None,
                 bank_seed: int = 11,
                 bank_replicates: int = 2,
                 T_decoder: int | None = None,
                 extractor: str = "identity",
                 prompt_mode: str = PRODUCER_PROMPT_NAIVE,
                 threshold_p95_ms: float = 1000.0,
                 threshold_error_rate: float = 0.10,
                 threshold_fw_count: int = 5,
                 replay_source: str = "",
                 verbose: bool = False,
                 ) -> dict[str, Any]:
    """Drive Phase 63 end-to-end.

    Pre-committed default config (R-63-baseline-loose sanity anchor):
    ``extractor='identity'``, ``prompt_mode='naive'``, ``T_decoder=None``,
    ``K_auditor=12``, ``T_auditor=256``, ``n_eval=8``, ``bank_seed=11``.

    The W16-1 anchor is reached by setting ``extractor='magnitude_filter'``,
    ``prompt_mode='structured'``, ``T_decoder=24``.
    """
    bank = _build_bank_for(extractor, n_replicates=bank_replicates,
                              bank_seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    extractor_obj = _build_extractor(
        extractor, prompt_mode=prompt_mode, seed=bank_seed,
        threshold_p95_ms=threshold_p95_ms,
        threshold_error_rate=threshold_error_rate,
        threshold_fw_count=threshold_fw_count,
        replay_source=replay_source)
    schemas = incident_triage_role_schemas()
    protocol = StructuredProducerProtocol(mode=prompt_mode)

    budgets = make_team_budgets(
        K_producer=K_producer, T_producer=T_producer,
        K_auditor=K_auditor, T_auditor=T_auditor)
    priorities = claim_priorities()

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    pack_stats_per_strategy: dict[str, list[dict[str, Any]]] = {
        s[0]: [] for s in _R63_STRATEGIES
    }
    for sc in bank:
        # Some extractors (Phase-61 mag-filter, replay) accept the
        # protocol/schema kwargs; Phase-62 identity does not. Dispatch
        # accordingly.
        if isinstance(extractor_obj, Phase62IdentityExtractor):
            round1_cands = extractor_obj.extract_round(sc, 1)
            round2_cands = extractor_obj.extract_round(sc, 2)
        else:
            recorded: list[ProducerPromptResult] = []
            round1_cands = extractor_obj.extract_round(
                sc, 1, protocol=protocol, schemas=schemas,
                record_prompts=recorded)
            round2_cands = extractor_obj.extract_round(
                sc, 2, protocol=protocol, schemas=schemas,
                record_prompts=recorded)
        bench_property_per_scenario[sc.scenario_id] = _bench_property_for(
            sc, round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in _R63_STRATEGIES:
            fac = _make_factory(sname, priorities, budgets)
            r, ps = _run_capsule_strategy(
                sc=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                T_decoder=T_decoder)
            results.append(r)
            if ps:
                pack_stats_per_strategy[sname].append({
                    "scenario_id": sc.scenario_id,
                    **ps,
                })

    strategy_names = ("substrate",) + tuple(s[0] for s in _R63_STRATEGIES)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        # The W16-1 strict-gain claim — composed wins over the
        # W14-only-budgeted baseline.
        "composed_minus_fifo_packed": gap(
            "capsule_attention_aware", "capsule_layered_fifo_packed"),
        # vs substrate FIFO:
        "composed_minus_fifo": gap(
            "capsule_attention_aware", "capsule_fifo"),
        # vs unpacked layered (no T_decoder enforcement; this is the
        # SDK v3.16 W15-3 backward-compat anchor when T_decoder=None):
        "composed_minus_layered": gap(
            "capsule_attention_aware", "capsule_layered_multi_round"),
        # The FIFO-packed-layered baseline (the failure-mode tracker):
        "fifo_packed_minus_layered": gap(
            "capsule_layered_fifo_packed",
            "capsule_layered_multi_round"),
        "max_non_attention_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_attention_aware"),
    }

    audit_ok_grid: dict[str, bool] = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    # The bench_summary keys depend on which family the bank uses.
    is_p62 = bool(bank and bank[0].scenario_id.startswith("p62_"))
    if is_p62:
        bench_summary = {
            "n_scenarios": len(bench_property_per_scenario),
            "scenarios_with_property": sum(
                1 for v in bench_property_per_scenario.values()
                if v.get("r62_property_holds")),
            "scenarios_with_multi_hypothesis": sum(
                1 for v in bench_property_per_scenario.values()
                if v.get("multi_hypothesis")),
            "scenarios_with_all_decoys_corr": sum(
                1 for v in bench_property_per_scenario.values()
                if v.get("all_decoys_round1_corroborated")),
            "scenarios_with_all_golds_single_role": sum(
                1 for v in bench_property_per_scenario.values()
                if v.get("all_golds_single_role")),
            "K_auditor": K_auditor,
            "T_decoder": T_decoder,
            "scenario_family": "p62_multi_hypothesis",
        }
    else:
        bench_summary = {
            "n_scenarios": len(bench_property_per_scenario),
            "scenarios_with_property": sum(
                1 for v in bench_property_per_scenario.values()
                if v.get("delayed_causal_evidence_property_holds")),
            "scenarios_with_decoy_corroboration": sum(
                1 for v in bench_property_per_scenario.values()
                if v.get("round1_decoy_corroborated")),
            "K_auditor": K_auditor,
            "T_decoder": T_decoder,
            "scenario_family": "p61_compmag",
        }

    def _agg_packstats(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {}
        n = len(rows)
        s_in = sum(r.get("tokens_input", 0) for r in rows)
        s_kept = sum(r.get("tokens_kept", 0) for r in rows)
        s_drop = sum(r.get("n_dropped_budget", 0) for r in rows)
        s_h_in = sum(r.get("n_handoffs_admitted", 0) for r in rows)
        s_h_kept = sum(r.get("n_handoffs_decoder_input", 0) for r in rows)
        pos = [r.get("position_of_first_causal_claim", -1) for r in rows]
        n_have = sum(1 for p in pos if p >= 0)
        avg_pos = (sum(p for p in pos if p >= 0) / max(1, n_have)
                    if n_have > 0 else -1.0)
        hin = [r.get("hypothesis_count_input", 0) for r in rows]
        hkept = [r.get("hypothesis_count_kept", 0) for r in rows]
        return {
            "n_cells": n,
            "tokens_input_sum": int(s_in),
            "tokens_kept_sum": int(s_kept),
            "n_dropped_budget_sum": int(s_drop),
            "tokens_kept_over_input": (round(s_kept / s_in, 4)
                                          if s_in > 0 else 0.0),
            "handoffs_admitted_sum": int(s_h_in),
            "handoffs_decoder_input_sum": int(s_h_kept),
            "fraction_handoffs_kept": (round(s_h_kept / s_h_in, 4)
                                          if s_h_in > 0 else 0.0),
            "position_of_first_causal_claim_avg": round(
                float(avg_pos), 4),
            "n_with_causal_claim_kept": int(n_have),
            "hypothesis_count_input_avg": round(
                sum(hin) / n, 4) if n else 0.0,
            "hypothesis_count_kept_avg": round(
                sum(hkept) / n, 4) if n else 0.0,
        }

    pack_stats_summary = {
        s: _agg_packstats(pack_stats_per_strategy.get(s, []))
        for s in ("capsule_layered_fifo_packed",
                   "capsule_attention_aware")
    }

    extractor_stats: dict[str, Any] = {
        "extractor": extractor,
        "prompt_mode": prompt_mode,
        "T_decoder": T_decoder,
    }
    if isinstance(extractor_obj, MagnitudeFilteringExtractor):
        extractor_stats["n_emitted"] = extractor_obj.n_emitted
        extractor_stats["n_filtered_by_threshold"] = (
            extractor_obj.n_filtered_by_threshold)
        extractor_stats["n_compressed_by_prompt"] = (
            extractor_obj.n_compressed_by_prompt)
        extractor_stats["n_emitted_under_protocol"] = (
            extractor_obj.n_emitted_under_protocol)
        extractor_stats["threshold_p95_ms"] = threshold_p95_ms
        extractor_stats["threshold_error_rate"] = threshold_error_rate
        extractor_stats["threshold_fw_count"] = threshold_fw_count
    if isinstance(extractor_obj, OllamaReplayExtractor):
        extractor_stats["replay_source"] = extractor_obj.source_path
        extractor_stats["n_replay_calls"] = extractor_obj.n_replay_calls
        extractor_stats["n_missing_keys"] = extractor_obj.n_missing_keys
        extractor_stats["recorded_prompt_mode"] = extractor_obj.prompt_mode

    if verbose:
        print(f"[phase63] extractor={extractor}, "
              f"prompt_mode={prompt_mode}, T_decoder={T_decoder}, "
              f"n_eval={len(bank)}, K_auditor={K_auditor}",
              file=sys.stderr, flush=True)
        print(f"[phase63] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase63]   {s:32s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase63] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)

    out: dict[str, Any] = {
        "schema": "phase63.composed_real_llm.v1",
        "config": {
            "n_eval": len(bank), "K_auditor": K_auditor,
            "T_auditor": T_auditor, "K_producer": K_producer,
            "T_producer": T_producer, "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "T_decoder": T_decoder,
            "extractor": extractor, "prompt_mode": prompt_mode,
            "threshold_p95_ms": threshold_p95_ms,
            "threshold_error_rate": threshold_error_rate,
            "threshold_fw_count": threshold_fw_count,
            "replay_source": replay_source,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "extractor_stats": extractor_stats,
        "pack_stats_summary": pack_stats_summary,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }
    return out


def run_phase63_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 8, K_auditor: int = 12, T_auditor: int = 256,
        T_decoder: int | None = 24,
        extractor: str = "magnitude_filter",
        prompt_mode: str = PRODUCER_PROMPT_STRUCTURED,
        ) -> dict[str, Any]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase63(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=2,
            T_decoder=T_decoder,
            extractor=extractor, prompt_mode=prompt_mode,
            verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
            "pack_stats_summary": rep["pack_stats_summary"],
        }
    return {
        "schema": "phase63.composed_real_llm_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor, "T_auditor": T_auditor,
        "T_decoder": T_decoder, "n_eval": n_eval,
        "extractor": extractor, "prompt_mode": prompt_mode,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 8, bank_seed: int = 11,
                                K_auditor: int = 12,
                                T_auditor: int = 256,
                                T_decoder_tight: int = 24,
                                T_decoder_degen: int = 2,
                                replay_source: str = "",
                                replay_naive_source: str = "",
                                T_decoder_replay_tight: int = 14,
                                K_auditor_replay: int = 8,
                                ) -> dict[str, Any]:
    """Six pre-committed sub-banks comparing the W16 cross-regime
    separation. Reports the headline accuracy per cell so the
    W14-Λ × W15-Λ composition / W16-1 / W16-2 / W16-3 separation is
    read off one table.

    The optional ``replay_source`` / ``replay_naive_source`` paths are
    Phase-61 real-Ollama reports (structured / naive prompt mode). The
    replay cells use ``T_decoder_replay_tight = 14`` by default — the
    pre-committed center of the budget band on Phase-61 single-decoy
    events where the FIFO pack drops the round-2 specific claim while
    the W15 salience pack keeps it (mechanically: unbroken bench
    property → ``capsule_attention_aware = 0.500`` and
    ``capsule_layered_fifo_packed = 0.000``; broken bench property →
    both at 0.000).
    """
    out: dict[str, Any] = {
        "schema": "phase63.cross_regime.v1",
        "config": {
            "n_eval": n_eval, "bank_seed": bank_seed,
            "K_auditor": K_auditor, "T_auditor": T_auditor,
            "T_decoder_tight": T_decoder_tight,
            "T_decoder_degen": T_decoder_degen,
            "replay_source": replay_source,
        },
    }
    # 1. Sanity baseline.
    out["r63_baseline_loose"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=None,
        extractor="identity", prompt_mode=PRODUCER_PROMPT_NAIVE,
        verbose=False)
    # 2. W15-only (= R-62-tightbudget on identity).
    out["r63_w15_only"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=T_decoder_tight,
        extractor="identity", prompt_mode=PRODUCER_PROMPT_NAIVE,
        verbose=False)
    # 3. W14-only (= bench broken upstream, no budget).
    out["r63_w14_only"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=None,
        extractor="magnitude_filter", prompt_mode=PRODUCER_PROMPT_NAIVE,
        verbose=False)
    # 4. naive + tight (W14-Λ × W15-Λ; everything fails).
    out["r63_naive_tight"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=T_decoder_tight,
        extractor="magnitude_filter", prompt_mode=PRODUCER_PROMPT_NAIVE,
        verbose=False)
    # 5. structured + loose (W14 alone restores property).
    out["r63_w14_success"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=None,
        extractor="magnitude_filter",
        prompt_mode=PRODUCER_PROMPT_STRUCTURED,
        verbose=False)
    # 6. structured + tight (W16-1 anchor).
    out["r63_composed_tight"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=T_decoder_tight,
        extractor="magnitude_filter",
        prompt_mode=PRODUCER_PROMPT_STRUCTURED,
        verbose=False)
    # 7. degenerate-budget falsifier.
    out["r63_degen_budget"] = run_phase63(
        n_eval=n_eval, bank_seed=bank_seed,
        K_auditor=K_auditor, T_auditor=T_auditor,
        T_decoder=T_decoder_degen,
        extractor="magnitude_filter",
        prompt_mode=PRODUCER_PROMPT_STRUCTURED,
        verbose=False)
    # 8. (optional) ollama_replay loose / composed-tight + naive-tight.
    # The replay path uses Phase-61 single-decoy events whose admitted
    # union token sum is smaller than Phase-62 multi-hypothesis. The
    # pre-committed budget band where W15 strictly beats FIFO-pack on
    # the recorded qwen2.5:14b-32k bytes is T_decoder ∈ [13, 16];
    # T_decoder_replay_tight = 14 is the centre of that band. Use
    # K_auditor_replay = 8 (Phase-61's anchor) so the admission
    # boundary matches the recorded probe exactly.
    if replay_source:
        out["r63_ollama_replay_loose"] = run_phase63(
            n_eval=n_eval, bank_seed=bank_seed,
            K_auditor=K_auditor_replay, T_auditor=T_auditor,
            T_decoder=None,
            extractor="ollama_replay",
            replay_source=replay_source,
            verbose=False)
        out["r63_ollama_replay_composed_tight"] = run_phase63(
            n_eval=n_eval, bank_seed=bank_seed,
            K_auditor=K_auditor_replay, T_auditor=T_auditor,
            T_decoder=T_decoder_replay_tight,
            extractor="ollama_replay",
            replay_source=replay_source,
            verbose=False)
    if replay_naive_source:
        out["r63_ollama_replay_naive_tight"] = run_phase63(
            n_eval=n_eval, bank_seed=bank_seed,
            K_auditor=K_auditor_replay, T_auditor=T_auditor,
            T_decoder=T_decoder_replay_tight,
            extractor="ollama_replay",
            replay_source=replay_naive_source,
            verbose=False)
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 63 — composed end-to-end W14+W15 benchmark "
                     "(SDK v3.17 / W16 family).")
    p.add_argument("--K-auditor", type=int, default=12)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--decoder-budget", type=int, default=-1,
                    help="Strict T_decoder budget. -1 = None (no "
                          "budget pressure).")
    p.add_argument("--extractor", type=str, default="identity",
                    choices=("identity", "magnitude_filter",
                             "ollama_replay"))
    p.add_argument("--prompt-mode", type=str,
                    default=PRODUCER_PROMPT_NAIVE,
                    choices=(PRODUCER_PROMPT_NAIVE,
                              PRODUCER_PROMPT_STRUCTURED))
    p.add_argument("--threshold-p95-ms", type=float, default=1000.0)
    p.add_argument("--threshold-error-rate", type=float, default=0.10)
    p.add_argument("--threshold-fw-count", type=int, default=5)
    p.add_argument("--replay-source", type=str, default="",
                    help="Path to a Phase-61 ollama report whose "
                          "raw_responses_per_scenario block this bench "
                          "should replay.")
    p.add_argument("--replay-naive-source", type=str, default="",
                    help="Path to a Phase-61 ollama-NAIVE report; "
                          "if set, the cross-regime adds the joint-failure "
                          "real-LLM cell.")
    p.add_argument("--cross-regime", action="store_true")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    T_dec = None if args.decoder_budget < 0 else int(args.decoder_budget)
    if args.cross_regime:
        report = run_cross_regime_summary(
            n_eval=args.n_eval, bank_seed=args.bank_seed,
            K_auditor=args.K_auditor, T_auditor=args.T_auditor,
            replay_source=args.replay_source,
            replay_naive_source=args.replay_naive_source)
    elif args.seed_sweep:
        report = run_phase63_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            T_decoder=(T_dec if T_dec is not None else 24),
            extractor=args.extractor, prompt_mode=args.prompt_mode)
    else:
        report = run_phase63(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            T_decoder=T_dec,
            extractor=args.extractor, prompt_mode=args.prompt_mode,
            threshold_p95_ms=args.threshold_p95_ms,
            threshold_error_rate=args.threshold_error_rate,
            threshold_fw_count=args.threshold_fw_count,
            replay_source=args.replay_source,
            verbose=not args.quiet)
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
            print(f"[phase63] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
