"""Phase 60 — open-world normalisation + real-Ollama transfer.

SDK v3.14, W13 family anchor. **First open-world normalisation move**
in the Wevra programme, and the **first real-Ollama benchmark with
honest raw-response capture**.

This phase has three pre-committed contributions:

1. **Wide-OOV synthetic regime (W13-1 anchor).** Extends Phase-59's
   :data:`OUT_OF_VOCAB_KINDS` set so the LLM emits drifted variants
   that are *outside* the closed-vocabulary :data:`CLAIM_KIND_SYNONYMS`
   table but *inside* the W13 heuristic abstraction closure
   (DEADLOCK_PROBABLY_DETECTED_MAYBE, POOL_LOOKING_BUSY,
   QUERY_SOMEWHAT_SLUGGISH, DISK_GETTING_FULL_PROBABLY, …). On this
   regime, the W12 fixed-table normaliser ties FIFO at 0.000 (the
   W12-4 collapse) while the W13
   :class:`LayeredRobustMultiRoundBundleDecoder` rescues the run.

2. **Truly arbitrary OOV regime (W13-4 anchor).** Inserts kinds that
   witness *neither* the exact synonym table *nor* any heuristic
   pattern (XYZZY_QQQQ, COSMIC_RAY_FLIP, INCIDENT_NUMBER_42,
   ABCDEF_GHIJK). On this regime, both W12 and W13 tie FIFO at
   0.000 — the named open-world closure boundary.

3. **Real-Ollama regime (R-60 anchor; honest raw-response capture).**
   Same Phase-58 delayed-causal-evidence shape but with a richer
   event stream designed so a real LLM at the 14B class actually
   produces the bench property. The driver records the raw producer
   string from every Ollama call into the report, plus per-role
   per-round drift counters, OOV rates, heuristic-rescue rates, and
   downstream correctness. The Phase-60 default reports honest
   numbers; if the LLM's drift is captured by the W13 closure, W13
   wins; if it stays canonical, W13 ties W12 at 1.000; if it emits
   true OOV outside both closures, W13 ties FIFO and we say so.

Theorem cross-reference (W13 family)
------------------------------------

* **W13-Λ-fixed** (proved-empirical + structural sketch). Fixed
  closed-vocabulary normalisation (:data:`CLAIM_KIND_SYNONYMS`) has
  a finite closure: any LLM kind variant outside the table survives
  normalisation unchanged. On a regime where the producer's drift
  channel emits ≥ 50% out-of-vocabulary kinds (Phase-60 wide-OOV
  bank), the W12 ``RobustMultiRoundBundleDecoder`` ties FIFO at
  0.000.
* **W13-1** (proved-conditional + proved-empirical). The
  :class:`LayeredRobustMultiRoundBundleDecoder` strictly widens the
  W12 closure: every variant in :data:`HEURISTIC_RESCUABLE_OOV_KINDS`
  is normalised to its canonical kind and the decoder achieves
  ``accuracy_full = 1.000`` while every un-widened method (W12
  included) ties FIFO at 0.000.
* **W13-2** (proved by inspection + mechanically-checked). Heuristic
  abstraction rule soundness: every rule's canonical output is in
  ``_DECODER_PRIORITY``; the heuristic layer is idempotent on
  canonical input; the heuristic layer never disagrees with the
  exact-table layer.
* **W13-3** (proved-empirical). Backward compatibility with R-58 /
  R-59 default / R-59-clean / R-54..R-57: on every prior anchor
  regime, the W13 decoder produces byte-for-byte the same answer
  as the W12 decoder; the heuristic layer fires zero times.
* **W13-4** (proved-empirical). Truly-arbitrary OOV falsifier: on
  the Phase-60 cosmic-OOV bank, W13 ties FIFO at 0.000 — the named
  open-world closure boundary. Expanding the heuristic table is a
  research move (W13-C2/W13-C3), not a structural fix.
* **W13-Λ-real** (empirical-research + observation). Real Ollama
  14B/35B on the Phase-58/60 prompt does not, by default, emit the
  delayed-causal-evidence bench property — it dedups events and
  emits canonical kinds. The synthetic-noisy extractor's drift
  channel is *narrower* than the synonym table but *wider* than the
  real LLM's drift channel on the calibrated prompt.

CLI
---

::

    # Default (synthetic_wide_oov, CI-runnable, W13-1 anchor):
    python3 -m vision_mvp.experiments.phase60_open_world_normalization \\
        --K-auditor 8 --n-eval 12 --out -

    # Cosmic-OOV falsifier (W13-4 anchor):
    python3 -m vision_mvp.experiments.phase60_open_world_normalization \\
        --falsifier --K-auditor 8 --n-eval 8 --out -

    # Backward-compat probe vs Phase-59 (W13-3 anchor):
    python3 -m vision_mvp.experiments.phase60_open_world_normalization \\
        --backward-compat --n-eval 8 --out -

    # 5-seed stability sweep:
    python3 -m vision_mvp.experiments.phase60_open_world_normalization \\
        --seed-sweep --n-eval 12 --out -

    # Real Ollama (R-60 honest baseline; opt-in):
    python3 -m vision_mvp.experiments.phase60_open_world_normalization \\
        --llm-mode ollama --endpoint http://192.168.12.191:11434 \\
        --model qwen2.5:14b-32k --n-eval 4 --out -
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
from vision_mvp.wevra.capsule import CapsuleKind, CapsuleLedger
from vision_mvp.wevra.llm_backend import LLMBackend, OllamaBackend
from vision_mvp.wevra.team_coord import (
    AdmissionPolicy, BundleAwareTeamDecoder,
    CLAIM_KIND_SYNONYMS,
    ClaimPriorityAdmissionPolicy,
    CohortCoherenceAdmissionPolicy,
    CoverageGuidedAdmissionPolicy,
    CrossRoleCorroborationAdmissionPolicy,
    FifoAdmissionPolicy,
    LayeredClaimNormalizer,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
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
    MultiRoundScenario, build_phase58_bank, _as_incident_scenario,
)
from vision_mvp.experiments.phase59_real_llm_multi_round import (
    NoisyLLMExtractor, NoisyLLMExtractorConfig, OllamaLLMExtractor,
    NOISY_KIND_VARIANTS, OUT_OF_VOCAB_KINDS,
    _round_ollama_prompt, _parse_ollama_response,
)


# =============================================================================
# Wide-OOV variant bank — heuristic-rescuable but not in fixed table
# =============================================================================
#
# These are LLM-style variants that are *outside* :data:`CLAIM_KIND_SYNONYMS`
# but *inside* the W13 heuristic abstraction closure. The Phase-60
# synthetic_wide_oov_llm extractor injects these with probability
# ``oov_prob`` (default 0.50) on causal claims. The W13 layered
# normaliser must rescue every entry; the W12 fixed-table normaliser
# cannot rescue any.

HEURISTIC_RESCUABLE_OOV_KINDS: dict[str, tuple[str, ...]] = {
    "DEADLOCK_SUSPECTED": (
        "DEADLOCK_PROBABLY_DETECTED_MAYBE",
        "DEADLOCK_LIKELY_OBSERVED",
        "DEADLOCK_CONDITION_DETECTED",
    ),
    "POOL_EXHAUSTION": (
        "POOL_LOOKING_BUSY",
        "POOL_AT_CAPACITY",
        "CONNECTION_POOL_FULL_NOW",
    ),
    "SLOW_QUERY_OBSERVED": (
        "QUERY_SOMEWHAT_SLUGGISH",
        "SLUGGISH_QUERIES_DETECTED",
        "QUERY_PERFORMANCE_SLOW",
    ),
    "DISK_FILL_CRITICAL": (
        "DISK_GETTING_FULL_PROBABLY",
        "DISK_AT_CAPACITY",
        "DISK_USAGE_CRITICAL_NOW",
    ),
    "ERROR_RATE_SPIKE": (
        "ERROR_RATE_HIGH",
        "ERROR_BURST",
        "ERROR_INCIDENT",
    ),
    "LATENCY_SPIKE": (
        "P95_REGRESSION",
        "SLO_VIOLATED",
        "LATENCY_INCREASE",
    ),
    "FW_BLOCK_SURGE": (
        "FW_DENIAL_BURST",
        "FIREWALL_REJECTED",
        "FW_DROP_SURGE",
    ),
}


# Mechanically-checked closure invariants (W13-1 / W13-2):
# * every entry is *not* in CLAIM_KIND_SYNONYMS (W12 cannot rescue)
# * normalising every entry through LayeredClaimNormalizer must yield
#   the canonical kind (W13-1 widens the closure)


# =============================================================================
# Cosmic-OOV variant bank — outside the W13 heuristic closure (W13-4)
# =============================================================================

COSMIC_OOV_KINDS: dict[str, tuple[str, ...]] = {
    "DEADLOCK_SUSPECTED": ("XYZZY_QQQQ", "COSMIC_RAY_FLIP"),
    "POOL_EXHAUSTION": ("INCIDENT_NUMBER_42", "FOO_BAR_BAZ"),
    "SLOW_QUERY_OBSERVED": ("ABCDEF_GHIJK", "PINEAPPLE_LANDED"),
    "DISK_FILL_CRITICAL": ("REDACTED_SIGNAL", "OPAQUE_TOKEN_X"),
}


# =============================================================================
# Phase-60 noisy extractor — adds wide-OOV + cosmic-OOV channels
# =============================================================================


@dataclasses.dataclass
class Phase60ExtractorConfig:
    """Phase-60 extractor knobs.

    Every drift channel is independently dialable. Default config is
    the W13-1 anchor: 50% wide-OOV (heuristic-rescuable) + 30% payload
    drift; no synonym drift (the LLM emits a *different* OOV variant,
    not the calibrated synonyms; W12 cannot rescue any of these).
    """
    synonym_prob: float = 0.0
    wide_oov_prob: float = 0.50
    cosmic_oov_prob: float = 0.0
    svc_token_alt_prob: float = 0.30
    drop_claim_prob: float = 0.0
    seed: int = 11

    def is_clean(self) -> bool:
        return (self.synonym_prob == 0.0
                and self.wide_oov_prob == 0.0
                and self.cosmic_oov_prob == 0.0
                and self.svc_token_alt_prob == 0.0
                and self.drop_claim_prob == 0.0)


def _phase60_drift_kind(canonical: str,
                          cfg: Phase60ExtractorConfig,
                          rng: random.Random) -> str | None:
    if cfg.drop_claim_prob > 0 and rng.random() < cfg.drop_claim_prob:
        return None
    if cfg.cosmic_oov_prob > 0 and rng.random() < cfg.cosmic_oov_prob:
        cosmic = COSMIC_OOV_KINDS.get(canonical, ())
        if cosmic:
            return cosmic[rng.randrange(0, len(cosmic))]
    if cfg.wide_oov_prob > 0 and rng.random() < cfg.wide_oov_prob:
        wide = HEURISTIC_RESCUABLE_OOV_KINDS.get(canonical, ())
        if wide:
            return wide[rng.randrange(0, len(wide))]
    if cfg.synonym_prob > 0 and rng.random() < cfg.synonym_prob:
        variants = NOISY_KIND_VARIANTS.get(canonical, ())
        if variants:
            return variants[rng.randrange(0, len(variants))]
    return canonical


def _phase60_drift_payload(payload: str,
                             cfg: Phase60ExtractorConfig,
                             rng: random.Random) -> str:
    if cfg.svc_token_alt_prob <= 0 or not payload:
        return payload
    from vision_mvp.experiments.phase59_real_llm_multi_round import (
        SERVICE_TAG_ALT_TEMPLATES)
    def _rewrite(m: re.Match) -> str:
        if rng.random() >= cfg.svc_token_alt_prob:
            return m.group(0)
        tag = m.group(1)
        return SERVICE_TAG_ALT_TEMPLATES[
            rng.randrange(0, len(SERVICE_TAG_ALT_TEMPLATES))].format(tag=tag)
    return re.sub(r"\bservice=([\w-]+)", _rewrite, payload)


@dataclasses.dataclass
class Phase60SyntheticExtractor:
    """Phase-60 synthetic extractor with wide-OOV and cosmic-OOV
    channels. Otherwise mirrors Phase-59's NoisyLLMExtractor."""

    cfg: Phase60ExtractorConfig

    def extract_round(self,
                       scenario: MultiRoundScenario,
                       round_idx: int,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        emissions = (scenario.round1_emissions if round_idx == 1
                       else scenario.round2_emissions)
        seed_key = (f"{self.cfg.seed}|{scenario.scenario_id}"
                     f"|round{round_idx}").encode("utf-8")
        seed_int = int.from_bytes(
            hashlib.sha256(seed_key).digest()[:8], "big")
        rng = random.Random(seed_int)
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            for (canonical_kind, canonical_payload) in emissions.get(
                    role, ()):
                drifted_kind = _phase60_drift_kind(canonical_kind,
                                                     self.cfg, rng)
                if drifted_kind is None:
                    continue
                drifted_payload = _phase60_drift_payload(
                    canonical_payload, self.cfg, rng)
                consumers = subs.consumers(role, canonical_kind)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, drifted_kind,
                                  drifted_payload, (0,)))
        return out


# =============================================================================
# Real-Ollama extractor with raw-response capture
# =============================================================================


@dataclasses.dataclass
class CapturingOllamaExtractor:
    """Real-Ollama extractor that records every raw producer string.

    Same wire shape as Phase-59's :class:`OllamaLLMExtractor` plus a
    ``raw_responses`` field keyed on ``(scenario_id, round_idx, role)``
    that the bench driver can dump into its forensic JSON."""

    backend: LLMBackend
    fallback_cfg: Phase60ExtractorConfig
    n_real_calls: int = 0
    n_failed_calls: int = 0
    total_wall_s: float = 0.0
    n_synthetic_fallbacks: int = 0
    raw_responses: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict)
    instruct_no_dedup: bool = True

    def _prompt(self,
                  role: str,
                  round_idx: int,
                  emissions: Sequence[tuple[str, str]],
                  allowed: Sequence[str]) -> str:
        # Reuse phase59 prompt, optionally adding a no-dedup hint.
        prompt = _round_ollama_prompt(role, round_idx, emissions, allowed)
        if self.instruct_no_dedup:
            prompt = prompt.replace(
                "Output rules: only KINDs from the list. One claim per line. "
                "Maximum 6 lines.",
                "Output rules: only KINDs from the list. EMIT ONE CLAIM PER "
                "DISTINCT EVENT BELOW; DO NOT SKIP OR DEDUPLICATE EVENTS EVEN "
                "IF THEY ARE SIMILAR. One claim per line. Maximum 8 lines.")
        return prompt

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
            prompt = self._prompt(role, round_idx, role_emissions, allowed)
            t0 = time.time()
            try:
                resp = self.backend.generate(
                    prompt, max_tokens=240, temperature=0.0)
                self.n_real_calls += 1
                self.total_wall_s += time.time() - t0
                self._record_raw(scenario.scenario_id, round_idx,
                                 role, resp,
                                 raw_kind="ollama_response",
                                 wall_s=time.time() - t0)
            except Exception as e:
                self.n_failed_calls += 1
                self.total_wall_s += time.time() - t0
                self._record_raw(scenario.scenario_id, round_idx,
                                 role, str(e),
                                 raw_kind="ollama_error",
                                 wall_s=time.time() - t0)
                fallback = Phase60SyntheticExtractor(self.fallback_cfg)
                role_out = [
                    e for e in fallback.extract_round(scenario, round_idx)
                    if e[0] == role]
                self.n_synthetic_fallbacks += 1
                out.extend(role_out)
                continue
            parsed = _parse_ollama_response(resp, allowed)
            for (kind, payload) in parsed:
                # Resolve consumers under either canonical or synonym.
                canon = CLAIM_KIND_SYNONYMS.get(kind, kind)
                consumers = subs.consumers(role, canon)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
        return out

    def _record_raw(self, scenario_id: str, round_idx: int, role: str,
                    text: str, *, raw_kind: str, wall_s: float) -> None:
        key = f"{scenario_id}|round{round_idx}|{role}"
        self.raw_responses[key] = {
            "scenario_id": scenario_id,
            "round_idx": int(round_idx),
            "role": role,
            "raw_kind": raw_kind,
            "text": text,
            "wall_s": round(wall_s, 3),
        }


# =============================================================================
# Phase-60 driver
# =============================================================================


def _make_factory(name: str, priorities, budgets):
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
        if name in ("capsule_multi_round",
                     "capsule_robust_multi_round",
                     "capsule_layered_multi_round"):
            return {r: FifoAdmissionPolicy() for r in budgets}
        raise ValueError(f"unknown strategy {name!r}")
    return fac


def _run_capsule_strategy(
        scenario: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        round2_cands: Sequence[tuple[str, str, str, str, tuple[int, ...]]],
        normaliser_metrics: list[dict[str, Any]] | None = None,
        ) -> StrategyResult:
    """Run two coordination rounds. ``decoder_mode`` is one of:
    per_round / single_round_bundle / multi_round_bundle /
    robust_multi_round / layered_multi_round (the new W13 mode).
    """
    incident_sc = _as_incident_scenario(scenario)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase60_open_world",
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

    if decoder_mode == "layered_multi_round":
        union = collect_admitted_handoffs(ledger, [rv1, rv2])
        decoder = LayeredRobustMultiRoundBundleDecoder()
        answer = decoder.decode_rounds([union])
        if normaliser_metrics is not None:
            normaliser_metrics.append({
                "scenario_id": scenario.scenario_id,
                "strategy": strategy_name,
                **decoder.normalizer_stats(),
            })
    elif decoder_mode == "robust_multi_round":
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
        round1_cands, round2_cands, inbox_capacity) -> StrategyResult:
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
        strategy="substrate", scenario_id=scenario.scenario_id,
        answer=answer, grading=grading, failure_kind=failure_kind,
        n_admitted_auditor=len(held),
        n_dropped_auditor_budget=auditor_inbox.n_overflow if auditor_inbox else 0,
        n_dropped_auditor_capacity=auditor_inbox.n_dedup if auditor_inbox else 0,
        n_dropped_auditor_unknown_kind=0,
        n_team_handoff=0, n_role_view=0, n_team_decision=0,
        audit_ok=False,
        n_tokens_admitted=sum(h.n_tokens for h in held),
    )


def _bench_property_post_normalisation(
        scenario: MultiRoundScenario,
        round1_cands, round2_cands,
        ) -> dict[str, Any]:
    """Verify the R-58 delayed-causal-evidence property holds after
    *layered* normalisation. Same shape as Phase-59's ``_bench_property``
    but using the W13 normaliser."""
    n = LayeredClaimNormalizer()
    GENERIC = frozenset({"LATENCY_SPIKE", "ERROR_RATE_SPIKE",
                          "FW_BLOCK_SURGE"})
    def _norm_kinds(cands):
        return {n.normalize(c[2]) for c in cands if c[1] == ROLE_AUDITOR}
    def _has_service(cands, decoy):
        for (_s, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            if f"service={decoy}" in normalize_payload(payload):
                return True
        return False
    def _decoy_roles(cands, decoy):
        roles = set()
        for (src, to, _k, payload, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            if f"service={decoy}" in normalize_payload(payload):
                roles.add(src)
        return roles
    r1_kinds = _norm_kinds(round1_cands)
    r2_kinds = _norm_kinds(round2_cands)
    decoy = scenario.decoy_storm_service
    r1_only_generic = bool(r1_kinds) and r1_kinds.issubset(GENERIC)
    r2_only_specific = bool(r2_kinds) and not (r2_kinds & GENERIC)
    r2_decoy = _has_service(round2_cands, decoy)
    r1_roles = _decoy_roles(round1_cands, decoy)
    return {
        "round1_only_generic_noise": r1_only_generic,
        "round2_only_specific": r2_only_specific,
        "decoy_only_in_round1": (not r2_decoy),
        "round1_decoy_corroborated": len(r1_roles) >= 2,
        "delayed_causal_evidence_property_holds": (
            r1_only_generic and r2_only_specific
            and (not r2_decoy) and len(r1_roles) >= 2),
        "n_round1_to_auditor": sum(
            1 for c in round1_cands if c[1] == ROLE_AUDITOR),
        "n_round2_to_auditor": sum(
            1 for c in round2_cands if c[1] == ROLE_AUDITOR),
    }


def _drift_metrics(round1_cands, round2_cands) -> dict[str, Any]:
    """Per-cell drift metrics. Counts handoffs whose pre-normalisation
    kind is NOT canonical, separated into:

      * ``n_w12_rescuable`` — in :data:`CLAIM_KIND_SYNONYMS` but not
        canonical (W12 rescues these);
      * ``n_w13_rescuable`` — in the W13 heuristic closure but not in
        the W12 table (only W13 rescues these);
      * ``n_oov`` — outside both closures (W13-4 falsifier territory).
    """
    canonical = {kind for (kind, _l, _r) in
                  __import__("vision_mvp.wevra.team_coord",
                              fromlist=["_DECODER_PRIORITY"])
                                ._DECODER_PRIORITY}
    layered = LayeredClaimNormalizer()
    n_w12 = n_w13 = n_oov = n_canonical = 0
    for cands in (round1_cands, round2_cands):
        for (_src, to, kind, _p, _e) in cands:
            if to != ROLE_AUDITOR:
                continue
            if kind in canonical:
                n_canonical += 1
                continue
            # Not canonical — what would normalisation do?
            in_synonym = kind.upper() in CLAIM_KIND_SYNONYMS
            if in_synonym:
                n_w12 += 1
                continue
            # Try W13 layered.
            layered.reset_counters()
            out = layered.normalize(kind)
            if out in canonical:
                n_w13 += 1
            else:
                n_oov += 1
    return {
        "n_canonical": n_canonical,
        "n_w12_rescuable": n_w12,
        "n_w13_rescuable": n_w13,
        "n_oov": n_oov,
    }


def run_phase60(*,
                  n_eval: int | None = None,
                  K_auditor: int = 8,
                  T_auditor: int = 256,
                  K_producer: int = 6,
                  T_producer: int = 96,
                  inbox_capacity: int | None = None,
                  bank_seed: int = 11,
                  bank_replicates: int = 3,
                  llm_mode: str = "synthetic_wide_oov_llm",
                  wide_oov_prob: float = 0.50,
                  cosmic_oov_prob: float = 0.0,
                  synonym_prob: float = 0.0,
                  svc_token_alt_prob: float = 0.30,
                  llm_seed: int = 11,
                  endpoint: str = "http://192.168.12.191:11434",
                  model: str = "qwen2.5:14b-32k",
                  timeout_s: float = 300.0,
                  capture_raw: bool = True,
                  verbose: bool = False,
                  ) -> dict[str, Any]:
    """Drive Phase 60 end-to-end.

    Default config (W13-1 anchor): ``llm_mode='synthetic_wide_oov_llm'``,
    ``wide_oov_prob=0.50``, ``svc_token_alt_prob=0.30``,
    ``K_auditor=8``, ``n_eval=12`` (4 builders × 3 replicates),
    ``bank_seed=11``.
    """
    bank = build_phase58_bank(n_replicates=bank_replicates, seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    cfg = Phase60ExtractorConfig(
        synonym_prob=synonym_prob,
        wide_oov_prob=wide_oov_prob,
        cosmic_oov_prob=cosmic_oov_prob,
        svc_token_alt_prob=svc_token_alt_prob,
        seed=llm_seed)

    extractor: Any
    if llm_mode == "synthetic_clean_llm":
        extractor = Phase60SyntheticExtractor(Phase60ExtractorConfig(
            synonym_prob=0.0, wide_oov_prob=0.0, cosmic_oov_prob=0.0,
            svc_token_alt_prob=0.0, seed=llm_seed))
    elif llm_mode == "synthetic_wide_oov_llm":
        extractor = Phase60SyntheticExtractor(cfg)
    elif llm_mode == "synthetic_cosmic_oov_llm":
        # Sharp W13-4 falsifier: every round-2 specific-tier disambiguating
        # claim is replaced by a truly arbitrary OOV token (cosmic_oov_prob=1.0
        # by default). The bench property still holds (the cosmic kinds are
        # not in the generic-noise set) but no normaliser can recover the
        # round-2 specific intent.
        extractor = Phase60SyntheticExtractor(Phase60ExtractorConfig(
            synonym_prob=0.0, wide_oov_prob=0.0,
            cosmic_oov_prob=max(cosmic_oov_prob, 1.0),
            svc_token_alt_prob=svc_token_alt_prob, seed=llm_seed))
    elif llm_mode == "synthetic_synonym_llm":
        # Phase-59 default — for backward-compat probe.
        extractor = Phase60SyntheticExtractor(Phase60ExtractorConfig(
            synonym_prob=0.50, wide_oov_prob=0.0, cosmic_oov_prob=0.0,
            svc_token_alt_prob=0.30, seed=llm_seed))
    elif llm_mode == "ollama":
        backend = OllamaBackend(
            model=model, base_url=endpoint, timeout=timeout_s,
            think=(False if "qwen3" in model else None))
        extractor = CapturingOllamaExtractor(
            backend=backend, fallback_cfg=cfg)
    else:
        raise ValueError(f"unknown llm_mode {llm_mode!r}")

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
        ("capsule_layered_multi_round", "layered_multi_round"),
    ]

    results: list[StrategyResult] = []
    bench_property_per_scenario: dict[str, dict[str, Any]] = {}
    drift_per_scenario: dict[str, dict[str, Any]] = {}
    normaliser_metrics: list[dict[str, Any]] = []
    raw_per_scenario: dict[str, list[dict[str, Any]]] = {}

    for sc in bank:
        round1_cands = extractor.extract_round(sc, 1)
        round2_cands = extractor.extract_round(sc, 2)
        bench_property_per_scenario[sc.scenario_id] = (
            _bench_property_post_normalisation(sc, round1_cands, round2_cands))
        drift_per_scenario[sc.scenario_id] = _drift_metrics(
            round1_cands, round2_cands)
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in strategies:
            fac = _make_factory(sname, priorities, budgets)
            results.append(_run_capsule_strategy(
                scenario=sc, budgets=budgets,
                policy_per_role_factory=fac,
                strategy_name=sname, decoder_mode=dmode,
                round1_cands=round1_cands,
                round2_cands=round2_cands,
                normaliser_metrics=normaliser_metrics
                    if sname == "capsule_layered_multi_round" else None,
            ))
        if (capture_raw and isinstance(extractor, CapturingOllamaExtractor)):
            raw_per_scenario[sc.scenario_id] = [
                v for k, v in extractor.raw_responses.items()
                if v["scenario_id"] == sc.scenario_id]

    strategy_names = ("substrate",) + tuple(s[0] for s in strategies)
    pooled = {s: pool(results, s).as_dict() for s in strategy_names}

    def gap(a: str, b: str) -> float:
        return round(pooled[a]["accuracy_full"]
                      - pooled[b]["accuracy_full"], 4)

    headline_gap = {
        "layered_minus_robust": gap(
            "capsule_layered_multi_round", "capsule_robust_multi_round"),
        "layered_minus_multi_round": gap(
            "capsule_layered_multi_round", "capsule_multi_round"),
        "layered_minus_fifo": gap(
            "capsule_layered_multi_round", "capsule_fifo"),
        "robust_minus_fifo": gap(
            "capsule_robust_multi_round", "capsule_fifo"),
        "robust_minus_multi_round": gap(
            "capsule_robust_multi_round", "capsule_multi_round"),
        "max_non_layered_accuracy_full": max(
            pooled[s]["accuracy_full"] for s in strategy_names
            if s != "capsule_layered_multi_round"),
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
    }
    drift_summary = {
        "n_canonical_total": sum(d["n_canonical"]
                                   for d in drift_per_scenario.values()),
        "n_w12_rescuable_total": sum(d["n_w12_rescuable"]
                                       for d in drift_per_scenario.values()),
        "n_w13_rescuable_total": sum(d["n_w13_rescuable"]
                                       for d in drift_per_scenario.values()),
        "n_oov_total": sum(d["n_oov"]
                              for d in drift_per_scenario.values()),
    }

    extractor_stats: dict[str, Any] = {
        "llm_mode": llm_mode,
        "noise_cfg": dataclasses.asdict(cfg)}
    if isinstance(extractor, CapturingOllamaExtractor):
        extractor_stats["n_real_calls"] = extractor.n_real_calls
        extractor_stats["n_failed_calls"] = extractor.n_failed_calls
        extractor_stats["total_wall_s"] = round(extractor.total_wall_s, 3)
        extractor_stats["n_synthetic_fallbacks"] = (
            extractor.n_synthetic_fallbacks)
        extractor_stats["instruct_no_dedup"] = bool(
            extractor.instruct_no_dedup)

    if verbose:
        print(f"[phase60] llm_mode={llm_mode}, n_eval={len(bank)}, "
              f"K_auditor={K_auditor}", file=sys.stderr, flush=True)
        print(f"[phase60] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank)}",
              file=sys.stderr, flush=True)
        print(f"[phase60] drift: canonical={drift_summary['n_canonical_total']} "
              f"w12_rescuable={drift_summary['n_w12_rescuable_total']} "
              f"w13_rescuable={drift_summary['n_w13_rescuable_total']} "
              f"oov={drift_summary['n_oov_total']}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase60]   {s:32s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase60] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)

    out = {
        "schema": "phase60.open_world_normalization.v1",
        "config": {
            "n_eval": len(bank), "K_auditor": K_auditor,
            "T_auditor": T_auditor,
            "K_producer": K_producer, "T_producer": T_producer,
            "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "llm_mode": llm_mode,
            "wide_oov_prob": wide_oov_prob,
            "cosmic_oov_prob": cosmic_oov_prob,
            "synonym_prob": synonym_prob,
            "svc_token_alt_prob": svc_token_alt_prob,
            "llm_seed": llm_seed,
            "endpoint": endpoint, "model": model,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "drift_summary": drift_summary,
        "drift_per_scenario": drift_per_scenario,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "extractor_stats": extractor_stats,
        "normaliser_metrics": normaliser_metrics,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }
    if raw_per_scenario:
        out["raw_responses_per_scenario"] = raw_per_scenario
    return out


def run_phase60_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 12, K_auditor: int = 8, T_auditor: int = 256,
        llm_mode: str = "synthetic_wide_oov_llm",
        wide_oov_prob: float = 0.50,
        ) -> dict[str, Any]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase60(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=3,
            llm_mode=llm_mode, wide_oov_prob=wide_oov_prob,
            llm_seed=seed, capture_raw=False, verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
            "drift_summary": rep["drift_summary"],
        }
    return {
        "schema": "phase60.open_world_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor, "T_auditor": T_auditor,
        "n_eval": n_eval,
        "llm_mode": llm_mode,
        "wide_oov_prob": wide_oov_prob,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 8, bank_seed: int = 11,
                                ) -> dict[str, Any]:
    """Single regime table: R-54..R-58 default + R-59 default + R-60
    wide-OOV / cosmic-OOV. The W13 backward-compat audit anchor."""
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
    from vision_mvp.experiments.phase59_real_llm_multi_round import (
        run_phase59)
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
    p59_noisy = run_phase59(n_eval=n_eval, K_auditor=8, T_auditor=256,
                              bank_seed=bank_seed, bank_replicates=2,
                              llm_mode="synthetic_noisy_llm",
                              verbose=False)
    p60_clean = run_phase60(n_eval=n_eval, K_auditor=8, T_auditor=256,
                              bank_seed=bank_seed, bank_replicates=2,
                              llm_mode="synthetic_clean_llm",
                              capture_raw=False, verbose=False)
    p60_wide = run_phase60(n_eval=n_eval, K_auditor=8, T_auditor=256,
                              bank_seed=bank_seed, bank_replicates=2,
                              llm_mode="synthetic_wide_oov_llm",
                              capture_raw=False, verbose=False)
    p60_cosmic = run_phase60(n_eval=n_eval, K_auditor=8, T_auditor=256,
                                bank_seed=bank_seed, bank_replicates=2,
                                llm_mode="synthetic_cosmic_oov_llm",
                                capture_raw=False, verbose=False)
    return {
        "schema": "phase60.cross_regime.v1",
        "config": {"n_eval": n_eval, "bank_seed": bank_seed},
        "phase54_default": p54,
        "phase55_default": p55,
        "phase56_default": p56,
        "phase57_default": p57,
        "phase58_default": p58,
        "phase59_noisy": p59_noisy,
        "phase60_clean": p60_clean,
        "phase60_wide_oov": p60_wide,
        "phase60_cosmic_oov": p60_cosmic,
    }


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 60 — open-world normalisation + real-Ollama "
                    "transfer (SDK v3.14 / W13 family).")
    p.add_argument("--K-auditor", type=int, default=8)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=12)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=3)
    p.add_argument("--falsifier", action="store_true")
    p.add_argument("--cross-regime", action="store_true")
    p.add_argument("--seed-sweep", action="store_true")
    p.add_argument("--backward-compat", action="store_true")
    p.add_argument("--llm-mode", type=str,
                    default="synthetic_wide_oov_llm",
                    choices=("synthetic_clean_llm",
                              "synthetic_wide_oov_llm",
                              "synthetic_cosmic_oov_llm",
                              "synthetic_synonym_llm",
                              "ollama"))
    p.add_argument("--wide-oov-prob", type=float, default=0.50)
    p.add_argument("--cosmic-oov-prob", type=float, default=0.0)
    p.add_argument("--synonym-prob", type=float, default=0.0)
    p.add_argument("--svc-alt-prob", type=float, default=0.30)
    p.add_argument("--llm-seed", type=int, default=11)
    p.add_argument("--endpoint", type=str,
                    default="http://192.168.12.191:11434")
    p.add_argument("--model", type=str, default="qwen2.5:14b-32k")
    p.add_argument("--timeout", type=float, default=300.0)
    p.add_argument("--no-capture-raw", action="store_true")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _arg_parser().parse_args(argv)
    if args.cross_regime:
        report = run_cross_regime_summary(
            n_eval=args.n_eval, bank_seed=args.bank_seed)
    elif args.seed_sweep:
        report = run_phase60_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            llm_mode=args.llm_mode,
            wide_oov_prob=args.wide_oov_prob)
    elif args.backward_compat:
        report = run_phase60(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            llm_mode="synthetic_clean_llm",
            capture_raw=False, verbose=not args.quiet)
    else:
        # Falsifier mode = cosmic OOV.
        if args.falsifier:
            llm_mode = "synthetic_cosmic_oov_llm"
        else:
            llm_mode = args.llm_mode
        report = run_phase60(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            llm_mode=llm_mode,
            wide_oov_prob=args.wide_oov_prob,
            cosmic_oov_prob=args.cosmic_oov_prob,
            synonym_prob=args.synonym_prob,
            svc_token_alt_prob=args.svc_alt_prob,
            llm_seed=args.llm_seed,
            endpoint=args.endpoint, model=args.model,
            timeout_s=args.timeout,
            capture_raw=not args.no_capture_raw,
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
            print(f"[phase60] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
