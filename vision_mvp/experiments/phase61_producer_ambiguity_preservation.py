"""Phase 61 — producer-side ambiguity preservation + structured prompt.

SDK v3.15, W14 family anchor. **First producer-protocol move** in
the Wevra programme — the direct response to the W13-Λ-real
honest negative observed on real Ollama 14B at SDK v3.14.

Recap of the W13-Λ-real diagnosis
=================================

Phase-60's ``--llm-mode ollama`` probe against
``qwen2.5:14b-32k`` on Mac 1 produced an honest negative: the LLM
emits *canonical* claim kinds (zero kind drift across n=4 × 12
producer calls) and *filters low-magnitude decoy events as noise*
(the ``monitor`` role emits ``NONE`` for the deliberately-low-
magnitude decoy events on the Phase-58 prompt). The bench property
holds in 0/4 scenarios; W12 / W13 normalisation has nothing to
rescue. The dominant blocker on real-LLM transfer is **producer-
side ambiguity erasure**, not normalisation.

Phase-61 attacks both root causes
=================================

1. **Comparable-magnitude decoy events.** The Phase-58 events have a
   deliberately-low-magnitude decoy (e.g. ``p95_ms=180`` while gold
   is ``p95_ms=2100``). A real LLM correctly filters these as noise.
   Phase-61 redesigns the events so the decoy storm service is a
   *real* collateral-damage service whose round-1 spikes are
   operationally indistinguishable from gold by magnitude alone.
   The disambiguation is recoverable *only* from the round-2 specific
   causal claim, not from magnitude. (R-61-default + R-61-naive +
   R-61-structured banks all use these events.)

2. **Structured producer protocol.** SDK v3.15's W14 contribution.
   The :class:`vision_mvp.wevra.team_coord.StructuredProducerProtocol`
   renders a prompt that splits round 1 (operational *observation*)
   from round 2 (specific *diagnosis*) and instructs the LLM to emit
   one claim per listed event with no compression. Phase-61's drivers
   wire this protocol into both the synthetic magnitude-filter
   extractor (so the synthetic bench gates the protocol cleanly) and
   the real-Ollama extractor (so the real-LLM probe can be graded on
   the R-61-OLLAMA 4-tier ladder).

Three pre-committed sub-banks plus an opt-in real-LLM extension
================================================================

* **R-61-default** (synthetic, no producer-side filtering). Same
  redesigned events; the synthetic extractor passes every event
  through to the candidate stream. Bench property holds in 8/8;
  every cross-round decoder (W11/W12/W13) hits 1.000. *Sanity
  anchor* — confirms the redesign does not break the synthetic
  property.

* **R-61-naive-prompt** (synthetic, magnitude-filter under naive
  prompt; W14-Λ-prompt anchor). The
  :class:`MagnitudeFilteringExtractor` simulates the observed
  real-LLM behaviour: events whose magnitude falls below a per-role
  threshold are dropped (emitted as ``NONE``); under the *naive*
  prompt, the extractor also collapses round-1 toward the highest-
  magnitude observation per service (mimicking the LLM's tendency
  to summarise). On the redesigned events, low-magnitude is no
  longer the issue (gold and decoy both pass the threshold) but
  the round-1 collapse is. The bench property holds in 0/8 by
  construction; every capsule strategy ties FIFO at 0.000. **The
  W14-Λ-prompt synthetic counterpart of W13-Λ-real.**

* **R-61-structured-prompt** (synthetic, magnitude-filter under
  structured protocol; W14-1 anchor). Same extractor wired to the
  structured prompt. The structured prompt's per-event mandate +
  observation-vs-diagnosis tier banner restores ``round1_decoy_
  corroborated`` on the synthetic side; W11/W12/W13 wins return.
  **The first synthetic bench in the programme that gates a
  producer-side intervention as a load-bearing layer.**

* **R-61-ollama** (opt-in real-LLM probe; W14-Λ-real anchor). Talks
  Mac-1 ``qwen2.5:14b-32k`` with the structured prompt. Honest
  4-tier grading mirrors R-60-OLLAMA-A..D — see
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` § 1.5.

Theorem cross-reference (W14 family)
====================================

* **W14-Λ-prompt** (proved-empirical + structural sketch) — on
  R-61-naive-prompt default, every capsule strategy in the SDK ties
  FIFO at 0.000. The bench property is erased upstream by the
  magnitude-filter extractor's round-1 compression; no downstream
  layer (admission / decoder / normaliser) can rescue an empty cross-
  role decoy corroboration set (extends the W7-3 extraction-floor
  argument: a never-emitted claim has no CID → no role view can
  admit it → no decoder can decode it).
* **W14-1** (proved-conditional + proved-empirical) — pairing the
  ``StructuredProducerProtocol`` with the same magnitude-filter
  extractor restores the bench property on R-61-structured-prompt;
  paired with the W13 layered decoder it achieves
  ``accuracy_full = 1.000`` on the same scenarios where the naive
  prompt ties FIFO at 0.000. The W14-1 win is *conditional* on the
  redesigned comparable-magnitude events (R-61-default property).
* **W14-2** (proved + mechanically-checked) — schema soundness:
  every role's ``observation_kinds`` ∪ ``diagnosis_kinds`` ⊆
  ``allowed_kinds``, the partition is disjoint on the incident-
  triage benchmark, and structured-prompt rendering is byte-for-
  byte deterministic given the schema + events.
* **W14-3** (proved-empirical) — backward compat with R-58 / R-59 /
  R-60: the structured prompt + the redesigned events do not break
  prior anchors; on R-61-default, every prior best capsule strategy
  hits ``accuracy_full = 1.000``; on R-58 / R-59-clean / R-60-clean,
  the legacy synthetic extractor (no magnitude filter) is unchanged.
* **W14-4** (proved-empirical) — without the structured protocol AND
  without the redesigned events, the W13-Λ-real diagnosis stands:
  Phase-60 Ollama with magnitude-filter-shaped events ties FIFO at
  0.250 on real Ollama. This is the named falsifier regime — the
  *combination* of comparable-magnitude events + structured prompt is
  the load-bearing intervention; either alone is insufficient.
* **W14-Λ-real** (empirical-research) — the *real-Ollama* outcome at
  Phase-61 default. Reported honestly per the R-61-OLLAMA tier ladder.

CLI
---

::

    # Default (R-61-default, synthetic, no-filter, CI-runnable):
    python3 -m vision_mvp.experiments.phase61_producer_ambiguity_preservation \\
        --K-auditor 8 --n-eval 8 --out -

    # R-61-naive-prompt (synthetic magnitude-filter + naive prompt;
    # W14-Λ-prompt anchor):
    python3 -m vision_mvp.experiments.phase61_producer_ambiguity_preservation \\
        --extractor magnitude_filter --prompt-mode naive --n-eval 8 --out -

    # R-61-structured-prompt (synthetic magnitude-filter + structured
    # prompt; W14-1 anchor):
    python3 -m vision_mvp.experiments.phase61_producer_ambiguity_preservation \\
        --extractor magnitude_filter --prompt-mode structured --n-eval 8 --out -

    # 5-seed stability sweep:
    python3 -m vision_mvp.experiments.phase61_producer_ambiguity_preservation \\
        --seed-sweep --n-eval 8 --out -

    # Real Ollama under structured prompt (R-61-OLLAMA, opt-in):
    python3 -m vision_mvp.experiments.phase61_producer_ambiguity_preservation \\
        --extractor ollama --prompt-mode structured \\
        --endpoint http://192.168.12.191:11434 --model qwen2.5:14b-32k \\
        --n-eval 4 --out -
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
    INCIDENT_TRIAGE_OBSERVATION_KINDS,
    LayeredClaimNormalizer,
    LayeredRobustMultiRoundBundleDecoder,
    MultiRoundBundleDecoder,
    MultiServiceCorroborationAdmissionPolicy,
    PRODUCER_PROMPT_MAGNITUDE_HINTED,
    PRODUCER_PROMPT_NAIVE, PRODUCER_PROMPT_STRUCTURED,
    ProducerPromptResult,
    RobustMultiRoundBundleDecoder, RoleBudget,
    RoleExtractionSchema,
    StructuredProducerProtocol,
    TeamCoordinator, audit_team_lifecycle,
    collect_admitted_handoffs, _DecodedHandoff,
    incident_triage_role_schemas,
    normalize_claim_kind, normalize_payload,
)
from vision_mvp.experiments.phase52_team_coord import (
    StrategyResult, _format_canonical_answer,
    claim_priorities, make_team_budgets, pool,
)
from vision_mvp.experiments.phase58_multi_round_decoder import (
    MultiRoundScenario, _as_incident_scenario,
)
from vision_mvp.experiments.phase59_real_llm_multi_round import (
    _parse_ollama_response,
)


# =============================================================================
# Structured-aware response parser (W14)
# =============================================================================
#
# The legacy ``_parse_ollama_response`` dedupes by ``claim_kind``, which
# is correct for the Phase-59 naive-prompt path (the LLM is asked to
# emit one claim of each kind it deems applicable; duplicates are LLM
# noise). Under the W14 structured protocol the LLM is *explicitly*
# instructed to emit one claim per listed event — and the events
# routinely share the same claim_kind across services. We therefore
# need a parser that dedupes by ``(kind, payload)`` rather than by
# ``kind`` alone, so the per-event mandate survives parsing.
#
# Backward compat: on naive-prompt outputs (where the model emits at
# most one claim per kind), this parser produces the same ``(kind,
# payload)`` set as the legacy path.

_STRUCTURED_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9_]*)\s*[|:\-–—]\s*(.+?)\s*$")


def _parse_structured_response(response: str,
                                  allowed: Sequence[str],
                                  max_claims: int = 16,
                                  ) -> list[tuple[str, str]]:
    """Per-event-aware closed-vocabulary parser for the W14 structured
    protocol. Dedupes by ``(kind, payload_normalised)`` rather than by
    ``kind`` alone — the structured prompt explicitly asks for one
    claim per listed event, so multiple claims of the same kind on
    *different* services must be preserved.
    """
    if not response:
        return []
    allowed_set = set(allowed)
    syn_set = set(CLAIM_KIND_SYNONYMS.keys())
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in response.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Lines that are *just* "NONE" or "NONE | <reason>" are skip
        # markers; the structured prompt allows them when no listed
        # event warrants any allowed kind.
        if line.upper().startswith("NONE"):
            continue
        m = _STRUCTURED_LINE_RE.match(line)
        if not m:
            continue
        kind = m.group(1).upper()
        payload = m.group(2).strip()[:240]
        if kind not in allowed_set and kind not in syn_set:
            continue
        # Dedupe by (kind, payload) — same-kind-different-service is
        # preserved; byte-identical lines collapse (Capsule Contract
        # C1 alignment).
        key = (kind, payload)
        if key in seen:
            continue
        seen.add(key)
        out.append((kind, payload))
        if len(out) >= max_claims:
            break
    return out


# =============================================================================
# Phase-61 redesigned scenario family — comparable-magnitude decoys
# =============================================================================
#
# Phase-58/59/60 uses deliberately-low-magnitude decoy events
# (``p95_ms=180`` while gold is ``p95_ms=2100``). A real LLM at the
# 14B class correctly filters these as noise (W13-Λ-real). Phase-61
# replaces the decoy magnitudes with values in the same operational
# range as gold so:
#
#   * a real LLM cannot use magnitude alone to discriminate;
#   * the disambiguation must come from the round-2 specific causal
#     claim AND the cross-round / cross-role corroboration logic
#     (the load-bearing capsule structure across W11/W12/W13);
#   * the synthetic ``MagnitudeFilteringExtractor`` becomes
#     informative — it can simulate "the LLM still discards events
#     it considers redundant" without merely simulating "the LLM
#     filters obvious noise".
#
# The decoy events are still *not* causally related to the gold
# root_cause; they represent collateral-damage spikes on a downstream
# service that just happened to coincide. The round-2 specific kind
# (DEADLOCK_SUSPECTED / POOL_EXHAUSTION / DISK_FILL_CRITICAL /
# SLOW_QUERY_OBSERVED) carries the actual cause.


# Comparable-magnitude operational ranges. Gold round-1 spikes use
# the high-magnitude tier (``p95_ms ∈ [1800, 4500]``,
# ``error_rate ∈ [0.12, 0.42]``); decoy round-1 spikes use the
# *same* tier with values close to but distinct from gold. The
# magnitude-filter extractor's threshold sits *below* both.
_GOLD_LATENCY_MS_RANGE = (1800, 4500)
_DECOY_LATENCY_MS_RANGE = (1700, 3200)  # overlapping with gold
_GOLD_ERROR_RATE_RANGE = (0.12, 0.42)
_DECOY_ERROR_RATE_RANGE = (0.10, 0.30)
_GOLD_FW_DENY_RANGE = (8, 18)
_DECOY_FW_DENY_RANGE = (6, 14)


def _emit(kind: str, payload: str) -> tuple[str, str]:
    return (kind, payload)


_REMEDIATION = {
    "deadlock":             "enforce_lock_ordering_in_orders",
    "pool_exhaustion":      "raise_pool_cap_or_fix_upstream",
    "slow_query_cascade":   "index_or_split_slow_query",
    "disk_fill":            "rotate_logs_and_clear_backup",
}


def _build_p61_deadlock(decoy: str = "search_index"
                          ) -> MultiRoundScenario:
    A, B = "orders", "payments"
    return MultiRoundScenario(
        scenario_id=(
            f"p61_deadlock_orders_payments__compmag_{decoy}"),
        description=(
            f"Phase-61: comparable-magnitude decoy. Round-1 generic "
            f"noise on gold {A}/{B} AND on cross-role-corroborated "
            f"{decoy} with operationally similar magnitudes. Round-2: "
            f"db_admin emits DEADLOCK_SUSPECTED (no service= token); "
            f"decoy gets nothing."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="deadlock",
        gold_remediation=_REMEDIATION["deadlock"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=2100 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.22 service={B}"),
                # Comparable-magnitude decoy spikes on monitor.
                _emit("LATENCY_SPIKE", f"p95_ms=1850 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.15 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=11 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("DEADLOCK_SUSPECTED",
                       f"deadlock relation={A}_{B}_join wait_chain=2"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_p61_pool(decoy: str = "archival") -> MultiRoundScenario:
    A, B = "api", "db"
    return MultiRoundScenario(
        scenario_id=f"p61_pool_api_db__compmag_{decoy}",
        description=(
            f"Phase-61: comparable-magnitude decoy. Round-1 generic "
            f"noise on gold {A}/{B} AND on {decoy} with operationally "
            f"similar magnitudes. Round-2: db_admin emits "
            f"POOL_EXHAUSTION (no service= token)."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="pool_exhaustion",
        gold_remediation=_REMEDIATION["pool_exhaustion"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4200 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.18 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=2400 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.13 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=12 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=14 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("POOL_EXHAUSTION",
                       "pool active=200/200 waiters=145 cluster=primary"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


def _build_p61_disk(decoy: str = "search_index"
                     ) -> MultiRoundScenario:
    A, B = "storage", "logs_pipeline"
    return MultiRoundScenario(
        scenario_id=f"p61_disk_storage_logs__compmag_{decoy}",
        description=(
            f"Phase-61: comparable-magnitude decoy. Round-1 generic "
            f"noise on gold {A}/{B} AND on {decoy} with operationally "
            f"similar magnitudes. Round-2: sysadmin emits "
            f"DISK_FILL_CRITICAL on the host (no service= token)."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="disk_fill",
        gold_remediation=_REMEDIATION["disk_fill"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.41 service={A}"),
                _emit("LATENCY_SPIKE", f"p95_ms=4500 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=3100 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.21 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=7 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=8 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_SYSADMIN: (
                _emit("DISK_FILL_CRITICAL",
                       "/var/log used=99% fs=/ host=primary"),
            ),
            ROLE_DB_ADMIN: (),
        },
    )


def _build_p61_slow_query(decoy: str = "metrics"
                            ) -> MultiRoundScenario:
    A, B = "web", "db"
    return MultiRoundScenario(
        scenario_id=f"p61_slow_query_web_db__compmag_{decoy}",
        description=(
            f"Phase-61: comparable-magnitude decoy. Round-1 generic "
            f"noise on gold {A}/{B} AND on {decoy} with operationally "
            f"similar magnitudes. Round-2: db_admin emits "
            f"SLOW_QUERY_OBSERVED."),
        gold_services_pair=(A, B),
        decoy_storm_service=decoy,
        gold_root_cause="slow_query_cascade",
        gold_remediation=_REMEDIATION["slow_query_cascade"],
        round1_emissions={
            ROLE_MONITOR: (
                _emit("LATENCY_SPIKE", f"p95_ms=4100 service={A}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.15 service={B}"),
                _emit("LATENCY_SPIKE", f"p95_ms=2900 service={decoy}"),
                _emit("ERROR_RATE_SPIKE",
                       f"error_rate=0.13 service={decoy}"),
            ),
            ROLE_NETWORK: (
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=9 service={decoy}"),
                _emit("FW_BLOCK_SURGE",
                       f"rule=deny count=10 service={decoy}"),
            ),
            ROLE_DB_ADMIN: (),
            ROLE_SYSADMIN: (),
        },
        round2_emissions={
            ROLE_MONITOR: (),
            ROLE_NETWORK: (),
            ROLE_DB_ADMIN: (
                _emit("SLOW_QUERY_OBSERVED",
                       "q#9 mean_ms=4100 cluster=primary"),
            ),
            ROLE_SYSADMIN: (),
        },
    )


_P61_BASE_BUILDERS = (
    _build_p61_deadlock, _build_p61_pool, _build_p61_disk,
    _build_p61_slow_query,
)
_P61_KNOWN_DECOYS = (
    "search_index", "archival", "metrics", "telemetry", "audit_jobs",
    "sessions", "cache", "scratch_pool",
)


def build_phase61_bank(*, n_replicates: int = 2,
                          seed: int = 11
                          ) -> list[MultiRoundScenario]:
    """Phase-61 default bank — comparable-magnitude decoy events.

    Mechanically-checkable closure invariants (W14-1 / W14-3):

    * For every scenario, the decoy magnitudes lie inside the gold
      magnitude range (verified by
      :func:`Phase61BankShapeTests.test_decoy_magnitudes_within_gold_range`).
    * Bench property holds in every scenario when the extractor
      passes events through unmodified (the R-61-default sanity
      anchor; verified by
      :func:`Phase61DefaultExtractorTests.test_bench_property_holds_8_of_8`).
    """
    rng = random.Random(seed)
    out: list[MultiRoundScenario] = []
    for builder in _P61_BASE_BUILDERS:
        for r in range(n_replicates):
            i = rng.randrange(0, 1 << 16)
            chosen = _P61_KNOWN_DECOYS[(i + r) % len(_P61_KNOWN_DECOYS)]
            sc = builder(decoy=chosen)
            out.append(dataclasses.replace(
                sc, scenario_id=f"{sc.scenario_id}__rep{r}"))
    return out


# =============================================================================
# Bench-property helpers — same shape as Phase-58/60, post-normalisation
# =============================================================================


_GENERIC_NOISE_AFTER_NORMALIZATION = frozenset(
    INCIDENT_TRIAGE_OBSERVATION_KINDS)


def _bench_property(sc: MultiRoundScenario,
                     round1_cands, round2_cands) -> dict[str, Any]:
    """Verify the R-58 delayed-causal-evidence property post-
    normalisation. Same shape as Phase-58/59/60's helper but threaded
    through the W13 layered normaliser so post-normalisation results
    are the comparison surface."""
    norm = LayeredClaimNormalizer()
    def _norm_kinds(cands):
        out = set()
        for c in cands:
            if c[1] != ROLE_AUDITOR:
                continue
            out.add(norm.normalize(c[2]))
        return out
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
    r1_kinds = _norm_kinds(round1_cands)
    r2_kinds = _norm_kinds(round2_cands)
    decoy = sc.decoy_storm_service
    round1_only_generic = bool(r1_kinds) and r1_kinds.issubset(
        _GENERIC_NOISE_AFTER_NORMALIZATION)
    round2_only_specific = bool(r2_kinds) and not (
        r2_kinds & _GENERIC_NOISE_AFTER_NORMALIZATION)
    r2_mentions_decoy = _has_service(round2_cands, decoy)
    r1_decoy_roles = _decoy_roles(round1_cands, decoy)
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
    }


# =============================================================================
# Identity (no-filter) extractor — the R-61-default sanity anchor
# =============================================================================


@dataclasses.dataclass
class IdentityExtractor:
    """No-filter, no-drift synthetic extractor. Passes every emission
    through to the candidate stream verbatim. The R-61-default sanity
    anchor — confirms the redesigned events satisfy the bench property
    when the producer does not filter."""

    seed: int = 11

    def extract_round(self,
                       sc: MultiRoundScenario,
                       round_idx: int,
                       *,
                       protocol: StructuredProducerProtocol | None = None,
                       schemas: dict[str, RoleExtractionSchema] | None = None,
                       record_prompts: list[ProducerPromptResult] | None = None,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        emissions = (sc.round1_emissions if round_idx == 1
                       else sc.round2_emissions)
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            role_em = emissions.get(role, ())
            if (role_em and protocol is not None and schemas
                    and role in schemas
                    and record_prompts is not None):
                # Render-and-record only: identity extractor does not
                # actually consult the protocol's output for routing.
                record_prompts.append(protocol.render_prompt(
                    role=role, round_idx=round_idx,
                    events=role_em, schema=schemas[role]))
            for (kind, payload) in role_em:
                consumers = subs.consumers(role, kind)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
        return out


# =============================================================================
# MagnitudeFilteringExtractor — synthetic simulation of W13-Λ-real
# =============================================================================
#
# Calibration target: the 14B Mac-1 Ollama probe in
# ``docs/data/phase60_real_ollama_qwen2_5_14b_n4.json``. The observed
# behaviour was:
#
#   * monitor sees 4 events (2 high-magnitude on gold + 2 low-
#     magnitude on decoy); emits 2 claims (both gold) + NONE × 2;
#   * network sees 2 events on decoy (rule=deny count=10..14); emits
#     2 claims on decoy;
#   * db_admin / sysadmin emit 1 specific claim on round 2.
#
# The synthetic magnitude-filter extractor reproduces this:
#
#   * The ``threshold`` knob filters out events whose embedded
#     magnitude falls below the threshold (Phase-58 low-magnitude
#     decoys → filtered; Phase-61 comparable-magnitude decoys →
#     preserved).
#   * The ``compress_round1_per_service`` knob simulates the LLM's
#     summarisation: under the *naive* prompt, the extractor keeps
#     only the highest-magnitude event per (role, service) pair.
#     Under the *structured* prompt, this knob is disabled — the
#     prompt explicitly mandates per-event extraction.


@dataclasses.dataclass
class MagnitudeFilteringExtractor:
    """Synthetic extractor calibrated against W13-Λ-real.

    Two knobs:

    * ``threshold_p95_ms`` — drop ``LATENCY_SPIKE`` emissions with
      ``p95_ms < threshold_p95_ms`` (default 1000, calibrated against
      the 14B observation).
    * ``threshold_error_rate`` — drop ``ERROR_RATE_SPIKE`` emissions
      with ``error_rate < threshold_error_rate`` (default 0.10).
    * ``threshold_fw_count`` — drop ``FW_BLOCK_SURGE`` emissions with
      ``count < threshold_fw_count`` (default 5).
    * ``prompt_mode`` — one of :data:`PRODUCER_PROMPT_NAIVE` or
      :data:`PRODUCER_PROMPT_STRUCTURED`. Under naive mode the
      extractor *also* compresses round-1 toward the highest-
      magnitude event per (role, service) (mimicking the real LLM's
      summarisation tendency on the naive prompt). Under structured
      mode this compression is disabled — the prompt's per-event
      mandate is the structural intervention.

    The extractor is deterministic given ``seed`` and the scenario_id
    (no RNG calls in normal operation; the seed is recorded for
    reproducibility provenance).
    """

    seed: int = 11
    threshold_p95_ms: float = 1000.0
    threshold_error_rate: float = 0.10
    threshold_fw_count: int = 5
    prompt_mode: str = PRODUCER_PROMPT_NAIVE
    # Under the naive prompt, the synthetic extractor mimics the
    # observed real-LLM compression by keeping only the top
    # ``naive_top_per_role_kind`` events per (role, claim_kind) bucket
    # (sorted by magnitude). The default 1 reproduces the W13-Λ-real
    # "the LLM reports each kind once per role even when multiple
    # events share that kind" behaviour. This is the load-bearing
    # synthetic counterpart of the real-LLM bottleneck.
    naive_top_per_role_kind: int = 1

    # Forensic counters populated per ``extract_round`` call.
    n_emitted: int = 0
    n_filtered_by_threshold: int = 0
    n_compressed_by_prompt: int = 0
    n_emitted_under_protocol: int = 0

    def reset_counters(self) -> None:
        self.n_emitted = 0
        self.n_filtered_by_threshold = 0
        self.n_compressed_by_prompt = 0
        self.n_emitted_under_protocol = 0

    def _passes_threshold(self, kind: str, payload: str) -> bool:
        if kind == "LATENCY_SPIKE":
            m = re.search(r"\bp95_ms=([0-9]+)", payload)
            if m:
                return int(m.group(1)) >= self.threshold_p95_ms
        if kind == "ERROR_RATE_SPIKE":
            m = re.search(r"\berror_rate=([0-9.]+)", payload)
            if m:
                return float(m.group(1)) >= self.threshold_error_rate
        if kind == "FW_BLOCK_SURGE":
            m = re.search(r"\bcount=([0-9]+)", payload)
            if m:
                return int(m.group(1)) >= self.threshold_fw_count
        return True  # other kinds always pass; only generic-noise tier
                     # filters by magnitude

    def _service_tag(self, payload: str) -> str:
        m = re.search(r"\bservice=([\w-]+)", payload)
        return m.group(1) if m else ""

    def _magnitude_score(self, kind: str, payload: str) -> float:
        if kind == "LATENCY_SPIKE":
            m = re.search(r"\bp95_ms=([0-9]+)", payload)
            return float(m.group(1)) if m else 0.0
        if kind == "ERROR_RATE_SPIKE":
            m = re.search(r"\berror_rate=([0-9.]+)", payload)
            return float(m.group(1)) * 10000.0 if m else 0.0
        if kind == "FW_BLOCK_SURGE":
            m = re.search(r"\bcount=([0-9]+)", payload)
            return float(m.group(1)) * 100.0 if m else 0.0
        return 0.0

    def extract_round(self,
                       sc: MultiRoundScenario,
                       round_idx: int,
                       *,
                       protocol: StructuredProducerProtocol | None = None,
                       schemas: dict[str, RoleExtractionSchema] | None = None,
                       record_prompts: list[ProducerPromptResult] | None = None,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        emissions = (sc.round1_emissions if round_idx == 1
                       else sc.round2_emissions)
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        prompt_mode = (protocol.mode if protocol is not None
                        else self.prompt_mode)
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            role_em = list(emissions.get(role, ()))
            if not role_em:
                continue
            if (protocol is not None and schemas and role in schemas
                    and record_prompts is not None):
                record_prompts.append(protocol.render_prompt(
                    role=role, round_idx=round_idx,
                    events=role_em, schema=schemas[role]))
            # Stage 1 — magnitude-filter every event.
            survivors: list[tuple[str, str]] = []
            for (kind, payload) in role_em:
                if self._passes_threshold(kind, payload):
                    survivors.append((kind, payload))
                else:
                    self.n_filtered_by_threshold += 1
            # Stage 2 — under naive prompt, compress to the top-N
            # events per (role, claim_kind) by magnitude. The default
            # N = 1 reproduces the W13-Λ-real "the LLM emits each kind
            # once per role even when multiple events share that
            # kind" behaviour: monitor sees {LATENCY on gold,
            # LATENCY on decoy, ERROR on gold, ERROR on decoy} and
            # emits one LATENCY (highest-magnitude → gold) plus one
            # ERROR (highest-magnitude → gold), dropping the decoy
            # observations. Under the structured prompt this stage
            # is disabled.
            if prompt_mode == PRODUCER_PROMPT_NAIVE and round_idx == 1:
                buckets: dict[str, list[tuple[float, str, str]]] = {}
                for (kind, payload) in survivors:
                    score = self._magnitude_score(kind, payload)
                    buckets.setdefault(kind, []).append(
                        (score, kind, payload))
                kept: list[tuple[str, str]] = []
                top_n = max(1, int(self.naive_top_per_role_kind))
                for kind, items in buckets.items():
                    items.sort(key=lambda x: (-x[0], x[2]))
                    for (_s, k, p) in items[:top_n]:
                        kept.append((k, p))
                self.n_compressed_by_prompt += (
                    len(survivors) - len(kept))
                survivors = kept
            elif (protocol is not None
                    and prompt_mode == PRODUCER_PROMPT_STRUCTURED):
                # Structured protocol: the per-event mandate is the
                # *prompt-side* contract; we count the number of
                # events the protocol asked the LLM to emit on so
                # the bench can verify the mandate fired.
                self.n_emitted_under_protocol += len(survivors)
            for (kind, payload) in survivors:
                consumers = subs.consumers(role, kind)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
                    self.n_emitted += 1
        return out


# =============================================================================
# Real Ollama extractor with structured-protocol support
# =============================================================================


@dataclasses.dataclass
class CapturingOllamaExtractor:
    """Real-Ollama extractor wired to the W14 producer protocol.

    Records every raw producer string into the report and exposes
    per-role, per-round drift / magnitude / compression metrics so the
    R-61-OLLAMA tier grading is mechanically verifiable.

    Falls back to a deterministic synthetic
    :class:`MagnitudeFilteringExtractor` on HTTP failure (the fallback
    is **labelled** in the report so the grading is honest about which
    cells used real-LLM extraction).
    """

    backend: LLMBackend
    fallback_cfg: dict[str, Any] = dataclasses.field(
        default_factory=dict)
    prompt_mode: str = PRODUCER_PROMPT_STRUCTURED
    n_real_calls: int = 0
    n_failed_calls: int = 0
    total_wall_s: float = 0.0
    n_synthetic_fallbacks: int = 0
    raw_responses: dict[str, dict[str, Any]] = dataclasses.field(
        default_factory=dict)

    def _record_raw(self, scenario_id: str, round_idx: int,
                     role: str, text: str, *, raw_kind: str,
                     wall_s: float, prompt_text: str,
                     prompt_mode: str) -> None:
        key = f"{scenario_id}|round{round_idx}|{role}"
        self.raw_responses[key] = {
            "scenario_id": scenario_id,
            "round_idx": int(round_idx),
            "role": role,
            "raw_kind": raw_kind,
            "text": text,
            "wall_s": round(wall_s, 3),
            "prompt_mode": prompt_mode,
            "prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")).hexdigest()[:16],
        }

    def extract_round(self,
                       sc: MultiRoundScenario,
                       round_idx: int,
                       *,
                       protocol: StructuredProducerProtocol | None = None,
                       schemas: dict[str, RoleExtractionSchema] | None = None,
                       record_prompts: list[ProducerPromptResult] | None = None,
                       ) -> list[tuple[str, str, str, str, tuple[int, ...]]]:
        if protocol is None:
            protocol = StructuredProducerProtocol(mode=self.prompt_mode)
        if schemas is None:
            schemas = incident_triage_role_schemas()
        emissions = (sc.round1_emissions if round_idx == 1
                       else sc.round2_emissions)
        subs = build_role_subscriptions()
        out: list[tuple[str, str, str, str, tuple[int, ...]]] = []
        for role in (ROLE_MONITOR, ROLE_DB_ADMIN, ROLE_SYSADMIN,
                       ROLE_NETWORK):
            role_em = list(emissions.get(role, ()))
            if not role_em:
                continue
            schema = schemas.get(role)
            if schema is None:
                continue
            prompt_res = protocol.render_prompt(
                role=role, round_idx=round_idx,
                events=role_em, schema=schema)
            if record_prompts is not None:
                record_prompts.append(prompt_res)
            t0 = time.time()
            try:
                resp = self.backend.generate(
                    prompt_res.text, max_tokens=320, temperature=0.0)
                self.n_real_calls += 1
                wall = time.time() - t0
                self.total_wall_s += wall
                self._record_raw(sc.scenario_id, round_idx, role, resp,
                                  raw_kind="ollama_response",
                                  wall_s=wall,
                                  prompt_text=prompt_res.text,
                                  prompt_mode=prompt_res.mode)
            except Exception as e:
                wall = time.time() - t0
                self.total_wall_s += wall
                self.n_failed_calls += 1
                self._record_raw(sc.scenario_id, round_idx, role,
                                  str(e), raw_kind="ollama_error",
                                  wall_s=wall,
                                  prompt_text=prompt_res.text,
                                  prompt_mode=prompt_res.mode)
                # Synthetic fallback (magnitude-filter under the same
                # prompt mode) so the run always completes.
                fb = MagnitudeFilteringExtractor(
                    prompt_mode=prompt_res.mode,
                    **self.fallback_cfg)
                role_out = [
                    e for e in fb.extract_round(sc, round_idx)
                    if e[0] == role]
                self.n_synthetic_fallbacks += 1
                out.extend(role_out)
                continue
            if prompt_res.mode in (PRODUCER_PROMPT_STRUCTURED,
                                     PRODUCER_PROMPT_MAGNITUDE_HINTED):
                parsed = _parse_structured_response(
                    resp, prompt_res.kinds_in_scope)
            else:
                parsed = _parse_ollama_response(
                    resp, prompt_res.kinds_in_scope)
            for (kind, payload) in parsed:
                canon = CLAIM_KIND_SYNONYMS.get(kind, kind)
                consumers = subs.consumers(role, canon)
                if not consumers:
                    continue
                for to_role in sorted(consumers):
                    out.append((role, to_role, kind, payload, (0,)))
        return out


# =============================================================================
# Phase-61 driver
# =============================================================================


def _build_extractor(extractor_kind: str, *,
                       prompt_mode: str,
                       seed: int,
                       endpoint: str,
                       model: str,
                       timeout_s: float,
                       threshold_p95_ms: float,
                       threshold_error_rate: float,
                       threshold_fw_count: int,
                       ):
    """Construct an extractor object from the named kind."""
    if extractor_kind == "identity":
        return IdentityExtractor(seed=seed)
    if extractor_kind == "magnitude_filter":
        return MagnitudeFilteringExtractor(
            seed=seed,
            threshold_p95_ms=threshold_p95_ms,
            threshold_error_rate=threshold_error_rate,
            threshold_fw_count=threshold_fw_count,
            prompt_mode=prompt_mode)
    if extractor_kind == "ollama":
        backend = OllamaBackend(
            model=model, base_url=endpoint, timeout=timeout_s,
            think=(False if "qwen3" in model else None))
        return CapturingOllamaExtractor(
            backend=backend,
            fallback_cfg={
                "seed": seed,
                "threshold_p95_ms": threshold_p95_ms,
                "threshold_error_rate": threshold_error_rate,
                "threshold_fw_count": threshold_fw_count,
            },
            prompt_mode=prompt_mode)
    raise ValueError(
        f"unknown extractor {extractor_kind!r}; "
        f"valid: identity / magnitude_filter / ollama")


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
        if name in ("capsule_multi_service", "capsule_bundle_decoder"):
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
        sc: MultiRoundScenario,
        budgets: dict[str, RoleBudget],
        policy_per_role_factory,
        strategy_name: str,
        *,
        decoder_mode: str,
        round1_cands, round2_cands,
        ) -> StrategyResult:
    incident_sc = _as_incident_scenario(sc)
    ledger = CapsuleLedger()
    coord = TeamCoordinator(
        ledger=ledger, role_budgets=budgets,
        policy_per_role=policy_per_role_factory(round_idx=1,
                                                  cands=round1_cands),
        team_tag="phase61_producer_protocol",
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
    return StrategyResult(
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


def run_phase61(*,
                 n_eval: int | None = None,
                 K_auditor: int = 8,
                 T_auditor: int = 256,
                 K_producer: int = 6,
                 T_producer: int = 96,
                 inbox_capacity: int | None = None,
                 bank_seed: int = 11,
                 bank_replicates: int = 2,
                 extractor: str = "identity",
                 prompt_mode: str = PRODUCER_PROMPT_NAIVE,
                 threshold_p95_ms: float = 1000.0,
                 threshold_error_rate: float = 0.10,
                 threshold_fw_count: int = 5,
                 endpoint: str = "http://192.168.12.191:11434",
                 model: str = "qwen2.5:14b-32k",
                 timeout_s: float = 300.0,
                 verbose: bool = False,
                 ) -> dict[str, Any]:
    """Drive Phase 61 end-to-end.

    Default config (R-61-default sanity anchor): ``extractor='identity'``,
    ``prompt_mode='naive'``, ``K_auditor=8``, ``n_eval=8`` (4 builders ×
    2 replicates), ``bank_seed=11``.
    """
    bank = build_phase61_bank(n_replicates=bank_replicates,
                                seed=bank_seed)
    if n_eval is not None:
        bank = bank[:n_eval]
    if inbox_capacity is None:
        inbox_capacity = K_auditor

    extractor_obj = _build_extractor(
        extractor, prompt_mode=prompt_mode, seed=bank_seed,
        endpoint=endpoint, model=model, timeout_s=timeout_s,
        threshold_p95_ms=threshold_p95_ms,
        threshold_error_rate=threshold_error_rate,
        threshold_fw_count=threshold_fw_count)
    schemas = incident_triage_role_schemas()
    protocol = StructuredProducerProtocol(mode=prompt_mode)

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
    prompt_records: list[dict[str, Any]] = []
    for sc in bank:
        recorded: list[ProducerPromptResult] = []
        round1_cands = extractor_obj.extract_round(
            sc, 1, protocol=protocol, schemas=schemas,
            record_prompts=recorded)
        round2_cands = extractor_obj.extract_round(
            sc, 2, protocol=protocol, schemas=schemas,
            record_prompts=recorded)
        bench_property_per_scenario[sc.scenario_id] = _bench_property(
            sc, round1_cands, round2_cands)
        # Record only the metadata of each prompt, not the full text;
        # the bench output already gets large.
        for pr in recorded:
            prompt_records.append({
                "scenario_id": sc.scenario_id,
                "role": pr.role,
                "round_idx": pr.round_idx,
                "mode": pr.mode,
                "n_kinds_in_scope": len(pr.kinds_in_scope),
                "prompt_sha256": hashlib.sha256(
                    pr.text.encode("utf-8")).hexdigest()[:16],
            })
        results.append(_run_substrate_strategy(
            sc, round1_cands, round2_cands, inbox_capacity))
        for (sname, dmode) in strategies:
            fac = _make_factory(sname, priorities, budgets)
            results.append(_run_capsule_strategy(
                sc=sc, budgets=budgets,
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
        "layered_minus_fifo": gap(
            "capsule_layered_multi_round", "capsule_fifo"),
        "layered_minus_robust": gap(
            "capsule_layered_multi_round", "capsule_robust_multi_round"),
        "layered_minus_multi_round": gap(
            "capsule_layered_multi_round", "capsule_multi_round"),
        "robust_minus_fifo": gap(
            "capsule_robust_multi_round", "capsule_fifo"),
        "robust_minus_multi_round": gap(
            "capsule_robust_multi_round", "capsule_multi_round"),
        "max_non_layered_accuracy_full": max(
            pooled[s]["accuracy_full"]
            for s in strategy_names if s != "capsule_layered_multi_round"),
    }

    audit_ok_grid = {}
    for s in strategy_names:
        if s == "substrate":
            audit_ok_grid[s] = False
            continue
        rs = [r for r in results if r.strategy == s]
        audit_ok_grid[s] = bool(rs) and all(r.audit_ok for r in rs)

    bench_summary = {
        "n_scenarios": len(bench_property_per_scenario),
        "scenarios_with_property": sum(
            1 for v in bench_property_per_scenario.values()
            if v["delayed_causal_evidence_property_holds"]),
        "scenarios_with_decoy_corroboration": sum(
            1 for v in bench_property_per_scenario.values()
            if v["round1_decoy_corroborated"]),
        "K_auditor": K_auditor,
    }

    extractor_stats: dict[str, Any] = {
        "extractor": extractor, "prompt_mode": prompt_mode,
        "threshold_p95_ms": threshold_p95_ms,
        "threshold_error_rate": threshold_error_rate,
        "threshold_fw_count": threshold_fw_count,
    }
    if isinstance(extractor_obj, MagnitudeFilteringExtractor):
        extractor_stats["n_emitted"] = extractor_obj.n_emitted
        extractor_stats["n_filtered_by_threshold"] = (
            extractor_obj.n_filtered_by_threshold)
        extractor_stats["n_compressed_by_prompt"] = (
            extractor_obj.n_compressed_by_prompt)
        extractor_stats["n_emitted_under_protocol"] = (
            extractor_obj.n_emitted_under_protocol)
    if isinstance(extractor_obj, CapturingOllamaExtractor):
        extractor_stats["n_real_calls"] = extractor_obj.n_real_calls
        extractor_stats["n_failed_calls"] = extractor_obj.n_failed_calls
        extractor_stats["n_synthetic_fallbacks"] = (
            extractor_obj.n_synthetic_fallbacks)
        extractor_stats["total_wall_s"] = round(
            extractor_obj.total_wall_s, 3)
        extractor_stats["endpoint"] = endpoint
        extractor_stats["model"] = model

    if verbose:
        print(f"[phase61] extractor={extractor}, "
              f"prompt_mode={prompt_mode}, n_eval={len(bank)}, "
              f"K_auditor={K_auditor}",
              file=sys.stderr, flush=True)
        print(f"[phase61] property holds in "
              f"{bench_summary['scenarios_with_property']}/{len(bank)}",
              file=sys.stderr, flush=True)
        print(f"[phase61] decoy_corroboration in "
              f"{bench_summary['scenarios_with_decoy_corroboration']}/{len(bank)}",
              file=sys.stderr, flush=True)
        for s in strategy_names:
            p = pooled[s]
            print(f"[phase61]   {s:32s} "
                  f"acc_full={p['accuracy_full']:.3f} "
                  f"acc_root_cause={p['accuracy_root_cause']:.3f} "
                  f"acc_services={p['accuracy_services']:.3f}",
                  file=sys.stderr, flush=True)
        for k, v in headline_gap.items():
            print(f"[phase61] {k}: {v:+.3f}",
                  file=sys.stderr, flush=True)

    out: dict[str, Any] = {
        "schema": "phase61.producer_ambiguity_preservation.v1",
        "config": {
            "n_eval": len(bank), "K_auditor": K_auditor,
            "T_auditor": T_auditor, "K_producer": K_producer,
            "T_producer": T_producer, "inbox_capacity": inbox_capacity,
            "bank_seed": bank_seed, "bank_replicates": bank_replicates,
            "extractor": extractor, "prompt_mode": prompt_mode,
            "threshold_p95_ms": threshold_p95_ms,
            "threshold_error_rate": threshold_error_rate,
            "threshold_fw_count": threshold_fw_count,
            "endpoint": endpoint, "model": model,
        },
        "pooled": pooled,
        "audit_ok_grid": audit_ok_grid,
        "bench_summary": bench_summary,
        "bench_property_per_scenario": bench_property_per_scenario,
        "headline_gap": headline_gap,
        "extractor_stats": extractor_stats,
        "prompt_records": prompt_records,
        "scenarios_evaluated": [sc.scenario_id for sc in bank],
        "n_results": len(results),
    }
    if isinstance(extractor_obj, CapturingOllamaExtractor):
        # Bucket raw responses by scenario for forensic capture.
        raw_per_sc: dict[str, list[dict[str, Any]]] = {}
        for entry in extractor_obj.raw_responses.values():
            raw_per_sc.setdefault(entry["scenario_id"], []).append(entry)
        out["raw_responses_per_scenario"] = raw_per_sc
    return out


def run_phase61_seed_stability_sweep(
        *, seeds: Sequence[int] = (11, 17, 23, 29, 31),
        n_eval: int = 8, K_auditor: int = 8, T_auditor: int = 256,
        extractor: str = "magnitude_filter",
        prompt_mode: str = PRODUCER_PROMPT_STRUCTURED,
        ) -> dict[str, Any]:
    per_seed: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        rep = run_phase61(
            n_eval=n_eval, K_auditor=K_auditor, T_auditor=T_auditor,
            bank_seed=seed, bank_replicates=2,
            extractor=extractor, prompt_mode=prompt_mode,
            verbose=False)
        per_seed[seed] = {
            "headline_gap": rep["headline_gap"],
            "pooled": rep["pooled"],
            "audit_ok_grid": rep["audit_ok_grid"],
            "bench_summary": rep["bench_summary"],
        }
    return {
        "schema": "phase61.producer_ambiguity_seed_sweep.v1",
        "seeds": list(seeds),
        "K_auditor": K_auditor, "T_auditor": T_auditor,
        "n_eval": n_eval,
        "extractor": extractor, "prompt_mode": prompt_mode,
        "per_seed": per_seed,
    }


def run_cross_regime_summary(*, n_eval: int = 8, bank_seed: int = 11,
                                ) -> dict[str, Any]:
    """Single regime table comparing R-61's three synthetic sub-banks.

    Reports the headline accuracy_full per (extractor × prompt_mode)
    cell so the W14-Λ-prompt / W14-1 / W14-3 separation is read off
    one table.
    """
    out: dict[str, Any] = {
        "schema": "phase61.cross_regime.v1",
        "config": {"n_eval": n_eval, "bank_seed": bank_seed},
    }
    out["r61_default_identity"] = run_phase61(
        n_eval=n_eval, bank_seed=bank_seed,
        extractor="identity", prompt_mode=PRODUCER_PROMPT_NAIVE,
        verbose=False)
    out["r61_naive_prompt"] = run_phase61(
        n_eval=n_eval, bank_seed=bank_seed,
        extractor="magnitude_filter",
        prompt_mode=PRODUCER_PROMPT_NAIVE,
        verbose=False)
    out["r61_structured_prompt"] = run_phase61(
        n_eval=n_eval, bank_seed=bank_seed,
        extractor="magnitude_filter",
        prompt_mode=PRODUCER_PROMPT_STRUCTURED,
        verbose=False)
    return out


# =============================================================================
# CLI
# =============================================================================


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 61 — producer-side ambiguity preservation "
                     "+ structured prompt (SDK v3.15 / W14 family).")
    p.add_argument("--K-auditor", type=int, default=8)
    p.add_argument("--T-auditor", type=int, default=256)
    p.add_argument("--n-eval", type=int, default=8)
    p.add_argument("--bank-seed", type=int, default=11)
    p.add_argument("--bank-replicates", type=int, default=2)
    p.add_argument("--extractor", type=str, default="identity",
                    choices=("identity", "magnitude_filter", "ollama"))
    p.add_argument("--prompt-mode", type=str,
                    default=PRODUCER_PROMPT_NAIVE,
                    choices=(PRODUCER_PROMPT_NAIVE,
                              PRODUCER_PROMPT_STRUCTURED))
    p.add_argument("--threshold-p95-ms", type=float, default=1000.0)
    p.add_argument("--threshold-error-rate", type=float, default=0.10)
    p.add_argument("--threshold-fw-count", type=int, default=5)
    p.add_argument("--cross-regime", action="store_true")
    p.add_argument("--seed-sweep", action="store_true")
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
        report = run_phase61_seed_stability_sweep(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor,
            extractor=args.extractor, prompt_mode=args.prompt_mode)
    else:
        report = run_phase61(
            n_eval=args.n_eval, K_auditor=args.K_auditor,
            T_auditor=args.T_auditor, bank_seed=args.bank_seed,
            bank_replicates=args.bank_replicates,
            extractor=args.extractor, prompt_mode=args.prompt_mode,
            threshold_p95_ms=args.threshold_p95_ms,
            threshold_error_rate=args.threshold_error_rate,
            threshold_fw_count=args.threshold_fw_count,
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
            print(f"[phase61] wrote {args.out}", file=sys.stderr)
    else:
        if not args.quiet:
            print(json.dumps(report.get("pooled", report),
                              indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
