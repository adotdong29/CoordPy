"""Phase 75 — ensemble-verified cross-model multi-chain pivot ratification
(SDK v3.29, W28 family).

The follow-up to SDK v3.28 (W27).  W27 routes each cell to a per-
signature W26 stack via a bounded pool of parallel chains keyed by the
salience signature CID.  It assumes the salience signature is *self-
evidently* the right routing key for that cell's content — any
byte-identical canonical state lands in the same chain, divergent
states route elsewhere.

W28 inserts a controller-side **ensemble ratification** step between
W27's signature lookup and its pool routing decision.  Before W27
commits to a pivot or new anchor, the controller polls a trust-
weighted probe table; the trust-weighted sum of ``ratify`` votes must
meet a pre-committed threshold for W27's routing to be ratified.

Phase-75 sub-banks
==================

Seven pre-committed sub-banks:

* **R-75-SINGLE-PROBE** (H2 anchor; W28-Λ-single-probe).
  K_probes=1 deterministic local-recompute probe.  W28 must reduce
  to W27 byte-for-byte across all cells.  Strict success bar:
  ``mean_total_w28_visible_tokens == mean_total_w27_visible_tokens``
  AND ``correctness_ratified_rate >= W27`` AND every cell ratified.

* **R-75-CHAIN-SHARED** (H2 anchor with multi-probe).
  K_probes=2 deterministic + oracle probes, single salience
  signature.  W28 must add at most 1 ratify-overhead token per cell
  (``wire_required = True`` from oracle probe).

* **R-75-CROSS-MODEL-DRIFT** (S3 / W28-3 headline; synthetic).
  A probe table of 3 probes: 2 deterministic (always ratify) +
  1 "drifting" probe that abstains on every other cell.  The trust-
  weighted quorum at threshold = 2.0 ratifies cells where ≥2 probes
  agree.  Variance vs W27 is tracked across seeds.

* **R-75-COORDINATED-DRIFT** (W28-Λ-coordinated-drift falsifier).
  A probe table of 3 probes that all use the same `drifting` source
  — every probe abstains on the same cells.  W28 cannot detect the
  drift; correctness ≤ W27.  Empirically confirmed by counting
  abstain rates per cell.

* **R-75-TRUST-ZERO** (W28-Λ-trust-zero falsifier).
  All probe trust priors set to 0.  Quorum is structurally
  unreachable; controller abstains on every cell.

* **R-75-RATIFICATION-TAMPERED** (H3 trust falsifier).
  After each ratification, a fraction of envelopes is tampered (the
  ``ratified`` flag flipped); the controller's verifier must reject
  with ``quorum_recompute_mismatch``.

* **R-75-POOL-EXHAUSTED** (W28-Λ-pool-exhausted-passthrough).
  ``max_active_chains = 2`` but the bench produces 4 distinct
  signatures; W28 must fall through to W27's pool-exhausted path
  (no spurious ratifications).

* **R-75-CROSS-HOST-LIVE** (S1/S2 best-effort; live LLM).
  Probe table includes :class:`LLMSignatureProbe` instances pointed
  at *two reachable Ollama hosts with different model families*
  (localhost + 192.168.12.191).  If both hosts are reachable,
  ``cross_host_round_trip_bytes > 0`` and the bench reports the
  cross-host probe call count.  If only one is reachable, the bench
  falls through gracefully to a single-host probe and labels the
  remaining gap.

The bench's apples-to-apples comparison is:

  * **W26 baseline**            — single ChainPersistedFanoutDisambiguator
  * **W27 orchestrator**        — pool of W26 stacks
  * **W28 ensemble-verified**   — W27 + ensemble ratification layer

Per-cell visible tokens, correctness, ratification rate, and cross-
host bytes are reported per configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Any

from vision_mvp.coordpy.team_coord import (
    OracleRegistration, SchemaCapsule,
    build_incident_triage_schema_capsule,
    SharedFanoutRegistry,
    # W26 / W27 (re-using phase74 stack)
    ChainPersistedFanoutRegistry,
    ChainPersistedFanoutDisambiguator,
    MultiChainPersistedFanoutOrchestrator,
    SharedMultiChainPool,
    compute_input_signature_cid,
    W27_BRANCH_PIVOTED, W27_BRANCH_ANCHORED_NEW,
    W27_BRANCH_POOL_EXHAUSTED, W27_BRANCH_PIVOT_REJECTED,
    W27_BRANCH_FALLBACK_W26, W27_BRANCH_NO_TRIGGER,
    W27_BRANCH_DISABLED,
    # W28 family
    ProbeVote,
    EnsembleProbe,
    EnsembleProbeRegistration,
    DeterministicSignatureProbe,
    OracleConsultationProbe,
    LLMSignatureProbe,
    EnsemblePivotRatificationEnvelope,
    EnsembleRatificationRegistry,
    EnsembleVerifiedMultiChainOrchestrator,
    W28EnsembleResult,
    verify_ensemble_pivot_ratification,
    build_default_ensemble_registry,
    build_two_probe_oracle_ensemble_registry,
    build_cross_host_llm_ensemble_registry,
    W28_RATIFICATION_SCHEMA_VERSION,
    W28_ALL_BRANCHES,
    W28_BRANCH_RATIFIED, W28_BRANCH_RATIFIED_PASSTHROUGH,
    W28_BRANCH_QUORUM_BELOW_THRESHOLD, W28_BRANCH_PROBE_REJECTED,
    W28_BRANCH_NO_RATIFY_NEEDED, W28_BRANCH_FALLBACK_W27,
    W28_BRANCH_NO_TRIGGER, W28_BRANCH_DISABLED,
    W28_DEFAULT_TRIGGER_BRANCHES,
    # Oracles
    ServiceGraphOracle, ChangeHistoryOracle,
    OutsideQuery,
    # Team handoff infrastructure
    _DecodedHandoff,
)
# Re-use the phase74 stack builders so W26/W27 baselines are
# byte-for-byte identical to the W27 milestone's bench.
from vision_mvp.experiments.phase74_multi_chain_pivot import (
    build_phase74_bank,
    build_team_shared_pool,
    build_team_shared_pool_xoracle,
    _build_w26_stack,
    _build_w27_orchestrator,
    _expected_gold_for_cell,
    _w24_visible, _w25_visible, _w26_visible, _w27_visible,
    _w27_branch, _w26_branch,
)

OLLAMA_LOCALHOST = os.environ.get(
    "COORDPY_OLLAMA_URL_LOCALHOST", "http://localhost:11434")
OLLAMA_MAC1 = os.environ.get(
    "COORDPY_OLLAMA_URL_MAC1", "http://192.168.12.191:11434")
OLLAMA_MAC2 = os.environ.get(
    "COORDPY_OLLAMA_URL_MAC2", "http://192.168.12.248:11434")

LOCAL_HOST_ID = socket.gethostname()


# ---------------------------------------------------------------------------
# Drift-aware synthetic probes (W28 family bench helpers)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class IntermittentDriftProbe:
    """A synthetic probe that abstains on every Nth cell.

    Used by R-75-CROSS-MODEL-DRIFT to model intermittent oracle
    disagreement in a deterministic (replayable) way. When
    ``cell_index % drift_period < drift_offset``, the probe abstains;
    otherwise it ratifies.
    """
    probe_id: str = "intermittent_drift"
    wire_required: bool = True
    drift_period: int = 3
    drift_offset: int = 1
    n_drifts_observed: int = 0
    n_ratifies: int = 0

    def vote(
            self,
            *,
            signature: Any,
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote:
        if int(cell_index) % int(self.drift_period) < int(self.drift_offset):
            self.n_drifts_observed += 1
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=False, reject=False,
                trust_weight=1.0, reason="intermittent_drift_abstain")
        self.n_ratifies += 1
        return ProbeVote(
            probe_id=self.probe_id,
            ratify=True, reject=False,
            trust_weight=1.0, reason="intermittent_drift_ratify")


@dataclasses.dataclass
class CoordinatedDriftProbe:
    """A synthetic probe sharing a drift_token with sibling probes;
    every probe with the same drift_token decides identically.

    Used by R-75-COORDINATED-DRIFT to confirm
    W28-Λ-coordinated-drift: when all probes are correlated, the
    ensemble cannot detect the drift.
    """
    probe_id: str = "coordinated_drift"
    wire_required: bool = True
    drift_token: int = 0  # cells where (cell_idx // period) % 2 == 0 ratify
    drift_period: int = 4
    n_ratifies: int = 0
    n_drifts: int = 0

    def vote(
            self,
            *,
            signature: Any,
            canonical_per_tag_votes: tuple[tuple[str, int], ...],
            canonical_projected_subset: tuple[str, ...],
            cell_index: int,
    ) -> ProbeVote:
        phase = (int(cell_index) // int(self.drift_period)) % 2
        if phase == int(self.drift_token):
            self.n_ratifies += 1
            return ProbeVote(
                probe_id=self.probe_id,
                ratify=True, reject=False,
                trust_weight=1.0, reason="coordinated_drift_ratify")
        self.n_drifts += 1
        return ProbeVote(
            probe_id=self.probe_id,
            ratify=False, reject=False,
            trust_weight=1.0, reason="coordinated_drift_abstain")


# ---------------------------------------------------------------------------
# Ensemble registry builders per sub-bank
# ---------------------------------------------------------------------------


def _ollama_reachable(url: str, *, timeout_s: float = 1.5) -> bool:
    """Best-effort probe: returns True iff /api/tags responds within
    ``timeout_s`` seconds with a JSON body containing a "models" key.
    """
    target = url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(target)
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read()
            payload = json.loads(body.decode("utf-8"))
            return isinstance(payload, dict) and "models" in payload
    except Exception:
        return False


def _list_ollama_models(url: str, *, timeout_s: float = 2.0) -> list[str]:
    target = url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(target)
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            body = resp.read()
            payload = json.loads(body.decode("utf-8"))
            return [str(m.get("model", "")) for m in payload.get("models", [])
                    if m.get("model")]
    except Exception:
        return []


def discover_two_host_topology() -> dict[str, Any]:
    """Probe the two-host Ollama topology and pick one model per host
    from *different model families* if possible.

    Returns a dict:
      {
        "topology": "two_host" | "single_host" | "unreachable",
        "hosts": [{"url": ..., "host_id": ..., "models": [...],
                   "selected_model": ...}, ...],
      }
    """
    candidates = [
        (OLLAMA_LOCALHOST, "localhost"),
        (OLLAMA_MAC1, "192.168.12.191"),
        (OLLAMA_MAC2, "192.168.12.248"),
    ]
    reached: list[dict[str, Any]] = []
    for url, host_id in candidates:
        if _ollama_reachable(url):
            models = _list_ollama_models(url)
            reached.append({
                "url": url, "host_id": host_id, "models": models,
            })
    if not reached:
        return {"topology": "unreachable", "hosts": []}

    # Architectural family from a model tag.  We prefer truly-different
    # families across hosts (e.g. Gemma2 + Qwen2.5, Llama3 + Qwen3, ...)
    # over two Qwen variants, so the cross-host ratification quorum
    # reflects real cross-architecture variance rather than two
    # close-relative models.
    def family(m: str) -> str:
        m = m.lower()
        if m.startswith("gemma"):
            return "gemma"
        if m.startswith("mixtral"):
            return "mixtral"
        if m.startswith("llama"):
            return "llama"
        if m.startswith("qwen3"):
            return "qwen3"
        if m.startswith("qwen2"):
            return "qwen2"
        if m.startswith("deepseek"):
            return "deepseek"
        return m.split(":")[0].split("-")[0].split("/")[-1]

    def family_priority(f: str) -> int:
        # Non-Qwen first when picking the *first* host's model so the
        # second host (likely 192.168.12.191 with only Qwen variants)
        # still gives architectural diversity.
        order = ["gemma", "mixtral", "llama", "deepseek",
                 "qwen3", "qwen2"]
        try:
            return order.index(f)
        except ValueError:
            return len(order)

    def _pick_model(host: dict, used_families: set[str]) -> tuple[str, str]:
        # Prefer a model whose family is not yet used; among the
        # remaining, prefer the family-priority order above.  Skip
        # tiny (sub-1B) AND huge (35B+) models so the live probe runs
        # in reasonable wall time on a single Ollama host.
        models = list(host["models"])
        models.sort(key=lambda m: (
            "0.5b" in m.lower(), "0.4b" in m.lower(),
            "35b" in m.lower(), "70b" in m.lower(),
            "32k" in m.lower(),
            family_priority(family(m))))
        for m in models:
            f = family(m)
            if f not in used_families:
                return m, f
        if models:
            m = models[0]
            return m, family(m)
        return "", ""

    if len(reached) >= 2:
        chosen: list[dict[str, Any]] = []
        used_families: set[str] = set()
        for h in reached:
            m, f = _pick_model(h, used_families)
            if not m:
                continue
            chosen.append({**h, "selected_model": m})
            used_families.add(f)
        if len(chosen) >= 2:
            return {"topology": "two_host", "hosts": chosen[:2]}
    h = reached[0]
    selected = h["models"][0] if h["models"] else ""
    return {
        "topology": "single_host",
        "hosts": [{**h, "selected_model": selected}],
    }


def build_ensemble_registry_for_bank(
        *,
        bank: str,
        schema: SchemaCapsule,
        local_host_id: str = "localhost",
) -> tuple[EnsembleRatificationRegistry, dict[str, Any]]:
    """Build the W28 registry for a given sub-bank.  Returns the
    registry plus a metadata dict with topology / probe info for the
    results record.
    """
    meta: dict[str, Any] = {"bank": bank, "topology": "synthetic"}

    if bank == "single_probe":
        return (
            build_default_ensemble_registry(
                schema=schema, quorum_threshold=1.0,
                local_host_id=local_host_id),
            {**meta, "n_probes": 1, "topology": "deterministic"},
        )

    if bank == "chain_shared":
        # K=2 deterministic + oracle probes on a single signature.
        sg = ServiceGraphOracle(oracle_id="service_graph_chain_shared")
        registry = EnsembleRatificationRegistry(
            schema=schema,
            quorum_threshold=2.0,
            probes=(
                EnsembleProbeRegistration(
                    probe=DeterministicSignatureProbe(
                        probe_id="local_recompute"),
                    trust_prior=1.0,
                    role_label="local_recompute",
                    host_id=local_host_id,
                ),
                EnsembleProbeRegistration(
                    probe=OracleConsultationProbe(
                        oracle=sg,
                        probe_id="oracle_service_graph"),
                    trust_prior=1.0,
                    role_label="service_graph",
                    host_id=local_host_id,
                ),
            ),
            local_host_id=local_host_id,
        )
        return registry, {**meta, "n_probes": 2, "topology": "two_probe_local"}

    if bank == "cross_model_drift":
        # 2 deterministic ratifies + 1 intermittent abstain.  Quorum=2.0
        # passes when both deterministic probes ratify (the intermittent
        # one abstains; trust-weighted weight = 2 ≥ 2.0).
        registry = EnsembleRatificationRegistry(
            schema=schema,
            quorum_threshold=2.0,
            probes=(
                EnsembleProbeRegistration(
                    probe=DeterministicSignatureProbe(
                        probe_id="local_recompute_a"),
                    trust_prior=1.0,
                    role_label="local_recompute_a",
                    host_id=local_host_id,
                ),
                EnsembleProbeRegistration(
                    probe=DeterministicSignatureProbe(
                        probe_id="local_recompute_b"),
                    trust_prior=1.0,
                    role_label="local_recompute_b",
                    host_id=local_host_id,
                ),
                EnsembleProbeRegistration(
                    probe=IntermittentDriftProbe(
                        probe_id="intermittent_drift",
                        drift_period=3, drift_offset=1),
                    trust_prior=1.0,
                    role_label="intermittent_drift",
                    host_id=local_host_id,
                ),
            ),
            local_host_id=local_host_id,
        )
        return registry, {**meta, "n_probes": 3, "topology": "synthetic_drift"}

    if bank == "coordinated_drift":
        # Three probes sharing the same drift_token — every probe
        # decides identically; ensemble cannot detect.
        registry = EnsembleRatificationRegistry(
            schema=schema,
            quorum_threshold=2.0,
            probes=(
                EnsembleProbeRegistration(
                    probe=CoordinatedDriftProbe(
                        probe_id="coord_drift_a",
                        drift_token=0, drift_period=4),
                    trust_prior=1.0,
                    role_label="coord_drift_a",
                    host_id=local_host_id,
                ),
                EnsembleProbeRegistration(
                    probe=CoordinatedDriftProbe(
                        probe_id="coord_drift_b",
                        drift_token=0, drift_period=4),
                    trust_prior=1.0,
                    role_label="coord_drift_b",
                    host_id=local_host_id,
                ),
                EnsembleProbeRegistration(
                    probe=CoordinatedDriftProbe(
                        probe_id="coord_drift_c",
                        drift_token=0, drift_period=4),
                    trust_prior=1.0,
                    role_label="coord_drift_c",
                    host_id=local_host_id,
                ),
            ),
            local_host_id=local_host_id,
        )
        return registry, {
            **meta, "n_probes": 3, "topology": "coordinated_drift"}

    if bank == "trust_zero":
        # All trust priors = 0; quorum unreachable; W28 always abstains.
        registry = EnsembleRatificationRegistry(
            schema=schema,
            quorum_threshold=1.0,
            probes=(
                EnsembleProbeRegistration(
                    probe=DeterministicSignatureProbe(
                        probe_id="zero_a"),
                    trust_prior=0.0,
                    role_label="zero_a",
                    host_id=local_host_id,
                ),
                EnsembleProbeRegistration(
                    probe=DeterministicSignatureProbe(
                        probe_id="zero_b"),
                    trust_prior=0.0,
                    role_label="zero_b",
                    host_id=local_host_id,
                ),
            ),
            local_host_id=local_host_id,
        )
        return registry, {**meta, "n_probes": 2, "topology": "trust_zero"}

    if bank == "ratification_tampered":
        # Same as chain_shared; the run loop tampers each envelope after
        # registration to confirm the verifier rejects.
        return build_ensemble_registry_for_bank(
            bank="chain_shared", schema=schema,
            local_host_id=local_host_id)

    if bank == "pool_exhausted":
        # Same registry as cross_model_drift; bench varies max_active_chains.
        return build_ensemble_registry_for_bank(
            bank="cross_model_drift", schema=schema,
            local_host_id=local_host_id)

    if bank == "cross_host_live":
        # Live LLM probe table — uses two reachable Ollama hosts.
        topo = discover_two_host_topology()
        meta["topology"] = topo["topology"]
        meta["hosts_used"] = [
            {"host_id": h["host_id"], "selected_model": h.get("selected_model", "")}
            for h in topo["hosts"]]
        if topo["topology"] == "unreachable":
            # Fall back to deterministic single probe so the bench still
            # runs; the results note will mark S1/S2 as unmet.
            return (
                build_default_ensemble_registry(
                    schema=schema, quorum_threshold=1.0,
                    local_host_id=local_host_id),
                {**meta, "n_probes": 1,
                  "topology_fallback": "deterministic_single"},
            )
        # Build LLM probes for every reachable host with a selected model.
        from vision_mvp.coordpy.llm_backend import OllamaBackend
        backends_with_hosts: list[tuple[Any, str, str, float]] = []
        for h in topo["hosts"]:
            if not h.get("selected_model"):
                continue
            backend = OllamaBackend(
                model=h["selected_model"],
                base_url=h["url"],
                timeout=10.0,
            )
            backends_with_hosts.append((
                backend, h["host_id"],
                f"{h['host_id']}_{h['selected_model']}",
                # Trust prior at 0.5 — LLMs are not as trusted as a
                # deterministic local recompute; production deployments
                # would calibrate via held-out agreement.
                0.5,
            ))
        # Always add a deterministic local probe so the quorum can be
        # met even if both LLM probes abstain (guarantees at least one
        # ratifying vote on byte-identical inputs).
        from vision_mvp.coordpy.team_coord import (
            DeterministicSignatureProbe as _D)
        registry = EnsembleRatificationRegistry(
            schema=schema,
            # Quorum threshold 1.0 = any single full-trust ratify
            # ratifies (deterministic local probe is trust 1.0).
            quorum_threshold=1.0,
            probes=(
                EnsembleProbeRegistration(
                    probe=_D(probe_id="local_recompute"),
                    trust_prior=1.0,
                    role_label="local_recompute",
                    host_id=local_host_id,
                ),
                *(
                    EnsembleProbeRegistration(
                        probe=LLMSignatureProbe(
                            backend=b, probe_id=f"llm_{role_label}"),
                        trust_prior=trust,
                        role_label=role_label,
                        host_id=host_id,
                    )
                    for b, host_id, role_label, trust in backends_with_hosts
                ),
            ),
            local_host_id=local_host_id,
        )
        meta["n_probes"] = 1 + len(backends_with_hosts)
        return registry, meta

    # Default — single deterministic probe.
    return (
        build_default_ensemble_registry(
            schema=schema, quorum_threshold=1.0,
            local_host_id=local_host_id),
        {**meta, "n_probes": 1, "topology": "default"},
    )


# ---------------------------------------------------------------------------
# Phase 75 runner
# ---------------------------------------------------------------------------


def _w28_visible(out: dict) -> int:
    if "ensemble_verified_multi_chain" in out:
        return int(out["ensemble_verified_multi_chain"].get(
            "n_w28_visible_tokens", 0))
    return _w27_visible(out)


def _w28_branch(out: dict) -> str:
    if "ensemble_verified_multi_chain" in out:
        return str(out["ensemble_verified_multi_chain"].get(
            "decoder_branch", ""))
    return ""


def _w28_ratified(out: dict) -> bool:
    if "ensemble_verified_multi_chain" in out:
        return bool(out["ensemble_verified_multi_chain"].get(
            "ratified", False))
    return False


def _build_w28_orchestrator(
        *,
        schema: SchemaCapsule,
        agent_id: str,
        is_producer: bool,
        producer_agent_id: str,
        consumer_agent_ids: tuple[str, ...],
        pool: SharedMultiChainPool,
        registry: EnsembleRatificationRegistry,
) -> EnsembleVerifiedMultiChainOrchestrator:
    """Wrap a W27 orchestrator with the W28 ensemble layer.

    Each agent gets its own W28 wrapper but shares the team-wide pool
    AND the team-wide ensemble registry (one ratification per cell,
    audited by every agent's controller).
    """
    inner = _build_w27_orchestrator(
        schema=schema,
        agent_id=str(agent_id),
        is_producer=bool(is_producer),
        producer_agent_id=str(producer_agent_id),
        consumer_agent_ids=tuple(consumer_agent_ids),
        pool=pool,
    )
    return EnsembleVerifiedMultiChainOrchestrator(
        inner=inner,
        registry=registry,
        enabled=True,
        require_ratification_verification=True,
        # Inherit the registry's local_host_id so cross-host telemetry
        # only fires when probes are *actually* on a different host.
        local_host_id=str(registry.local_host_id),
    )


def run_phase75(
        *,
        bank: str = "cross_model_drift",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run one Phase-75 sub-bank and return the results dict.

    Compares THREE configurations on the same bench:

    * **W26 baseline**          — single ChainPersistedFanoutDisambiguator
    * **W27 orchestrator**      — pool of W26 stacks (no ensemble layer)
    * **W28 ensemble-verified** — W27 + EnsembleVerifiedMultiChainOrchestrator

    Tokens, correctness, and ratification rate are reported per
    configuration.
    """
    schema = build_incident_triage_schema_capsule()

    # Map sub-bank to underlying phase74 bank shape.
    if bank in ("cross_model_drift", "coordinated_drift", "trust_zero",
                "ratification_tampered", "cross_host_live"):
        underlying_bank = "divergent_recover"
    elif bank == "pool_exhausted":
        underlying_bank = "pool_exhausted"
    elif bank == "single_probe":
        underlying_bank = "chain_shared"
    elif bank == "chain_shared":
        underlying_bank = "chain_shared"
    else:
        underlying_bank = "divergent_recover"

    raw_oracles: tuple[tuple[Any, str], ...] = (
        (ServiceGraphOracle(oracle_id="service_graph"), "service_graph"),
        (ChangeHistoryOracle(oracle_id="change_history"), "change_history"),
    )

    cells = build_phase74_bank(
        n_replicates=bank_replicates, seed=bank_seed,
        n_cells=n_eval, bank=underlying_bank,
        signature_period=signature_period)

    producer_id = "producer_agent"
    consumer_ids = tuple(f"consumer_{k}" for k in range(K_consumers))

    if chain_persist_window is None:
        chain_persist_window = n_eval

    effective_max_chains = (
        2 if bank == "pool_exhausted" else max_active_chains)

    projection_id_for_consumer = {
        cid: f"proj_{cid}" for cid in consumer_ids
    }
    projected_tags_for_consumer = {
        cid: ("orders", "payments", "api", "db",
              "storage", "logs_pipeline", "web", "db_query")
        for cid in consumer_ids
    }

    # ---------- W26 baseline ----------
    fanout_registry_w26 = SharedFanoutRegistry(schema=schema)
    chain_registry_w26 = ChainPersistedFanoutRegistry(schema=schema)
    producer_w26 = _build_w26_stack(
        T_decoder=T_decoder, schema=schema, raw_oracles=raw_oracles,
        agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        fanout_registry=fanout_registry_w26,
        chain_registry=chain_registry_w26,
        chain_persist_window=chain_persist_window,
        projection_id_for_consumer=projection_id_for_consumer,
        projected_tags_for_consumer=projected_tags_for_consumer,
    )
    consumer_w26_list = [
        _build_w26_stack(
            T_decoder=T_decoder, schema=schema,
            raw_oracles=raw_oracles,
            agent_id=cid, is_producer=False,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids,
            fanout_registry=fanout_registry_w26,
            chain_registry=chain_registry_w26,
            chain_persist_window=chain_persist_window,
            projection_id_for_consumer=projection_id_for_consumer,
            projected_tags_for_consumer=projected_tags_for_consumer,
        )
        for cid in consumer_ids
    ]

    # ---------- W27 orchestrator (no ensemble) ----------
    pool_w27 = build_team_shared_pool(
        T_decoder=T_decoder, schema=schema,
        raw_oracles=raw_oracles,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        chain_persist_window=chain_persist_window,
        max_active_chains=effective_max_chains,
        projection_id_for_consumer=projection_id_for_consumer,
        projected_tags_for_consumer=projected_tags_for_consumer,
    )
    producer_w27 = _build_w27_orchestrator(
        schema=schema, agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids, pool=pool_w27)
    consumer_w27_list = [
        _build_w27_orchestrator(
            schema=schema, agent_id=cid, is_producer=False,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids, pool=pool_w27)
        for cid in consumer_ids
    ]

    # ---------- W28 ensemble-verified orchestrator ----------
    registry_w28, registry_meta = build_ensemble_registry_for_bank(
        bank=bank, schema=schema, local_host_id=LOCAL_HOST_ID)
    pool_w28 = build_team_shared_pool(
        T_decoder=T_decoder, schema=schema,
        raw_oracles=raw_oracles,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids,
        chain_persist_window=chain_persist_window,
        max_active_chains=effective_max_chains,
        projection_id_for_consumer=projection_id_for_consumer,
        projected_tags_for_consumer=projected_tags_for_consumer,
    )
    producer_w28 = _build_w28_orchestrator(
        schema=schema, agent_id=producer_id, is_producer=True,
        producer_agent_id=producer_id,
        consumer_agent_ids=consumer_ids, pool=pool_w28,
        registry=registry_w28)
    consumer_w28_list = [
        _build_w28_orchestrator(
            schema=schema, agent_id=cid, is_producer=False,
            producer_agent_id=producer_id,
            consumer_agent_ids=consumer_ids, pool=pool_w28,
            registry=registry_w28)
        for cid in consumer_ids
    ]

    per_cell_producer_w26: list[dict[str, Any]] = []
    per_cell_producer_w27: list[dict[str, Any]] = []
    per_cell_producer_w28: list[dict[str, Any]] = []
    per_cell_consumers_w26: list[list[dict[str, Any]]] = []
    per_cell_consumers_w27: list[list[dict[str, Any]]] = []
    per_cell_consumers_w28: list[list[dict[str, Any]]] = []
    correctness_w26: list[bool] = []
    correctness_w27: list[bool] = []
    correctness_w28: list[bool] = []
    n_tampered_rejected: int = 0
    n_tamper_attempts: int = 0

    for cell_idx, cell_handoffs in enumerate(cells):
        p_out_w26 = producer_w26.decode_rounds(cell_handoffs)
        per_cell_producer_w26.append(p_out_w26)
        p_out_w27 = producer_w27.decode_rounds(cell_handoffs)
        per_cell_producer_w27.append(p_out_w27)
        p_out_w28 = producer_w28.decode_rounds(cell_handoffs)
        per_cell_producer_w28.append(p_out_w28)

        # Tampering bank: tamper the just-recorded ratification envelope
        # AFTER it was registered, then verify that *re-verifying* it
        # against the registry rejects the tampered form.
        if bank == "ratification_tampered":
            envelope = producer_w28.last_envelope
            if envelope is not None:
                # Flip the ratified flag and recompute (broken) cid.
                tampered = dataclasses.replace(
                    envelope, ratified=(not envelope.ratified),
                    ratification_cid="")
                outcome = verify_ensemble_pivot_ratification(
                    tampered,
                    registered_schema=schema,
                    registered_signature_cid=envelope.signature_cid,
                    registered_probe_ids=registry_w28.registered_probe_ids,
                )
                n_tamper_attempts += 1
                if not outcome.ok:
                    n_tampered_rejected += 1

        expected = _expected_gold_for_cell(
            bank=underlying_bank, cell_idx=cell_idx, n_eval=n_eval,
            signature_period=signature_period)

        def _is_correct(out: dict) -> bool:
            ans = out.get("answer") or out
            svcs = ans.get("services") if isinstance(ans, dict) else None
            if svcs is None:
                svcs = out.get("services")
            return set(svcs or []) == expected

        correctness_w26.append(_is_correct(p_out_w26))
        correctness_w27.append(_is_correct(p_out_w27))
        correctness_w28.append(_is_correct(p_out_w28))

        # Consumers
        c_row_w26 = [c.decode_rounds(cell_handoffs)
                       for c in consumer_w26_list]
        per_cell_consumers_w26.append(c_row_w26)
        c_row_w27 = [c.decode_rounds(cell_handoffs)
                       for c in consumer_w27_list]
        per_cell_consumers_w27.append(c_row_w27)
        c_row_w28 = [c.decode_rounds(cell_handoffs)
                       for c in consumer_w28_list]
        per_cell_consumers_w28.append(c_row_w28)

    n_cells_run = len(per_cell_producer_w26)

    # Token accounting.
    w26_tokens_p_b = [_w26_visible(o) for o in per_cell_producer_w26]
    w27_tokens_p = [_w27_visible(o) for o in per_cell_producer_w27]
    w28_tokens_p = [_w28_visible(o) for o in per_cell_producer_w28]

    w26_tokens_c_b = [
        sum(_w26_visible(c) for c in row) for row in per_cell_consumers_w26]
    w27_tokens_c = [
        sum(_w27_visible(c) for c in row) for row in per_cell_consumers_w27]
    w28_tokens_c = [
        sum(_w28_visible(c) for c in row) for row in per_cell_consumers_w28]

    total_w26 = [w26_tokens_p_b[i] + w26_tokens_c_b[i]
                  for i in range(n_cells_run)]
    total_w27 = [w27_tokens_p[i] + w27_tokens_c[i]
                  for i in range(n_cells_run)]
    total_w28 = [w28_tokens_p[i] + w28_tokens_c[i]
                  for i in range(n_cells_run)]

    def _mean(xs: list[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _stddev(xs: list[int]) -> float:
        n = len(xs)
        if n < 2:
            return 0.0
        m = _mean(xs)
        return (sum((x - m) ** 2 for x in xs) / (n - 1)) ** 0.5

    mean_w26 = _mean(total_w26)
    mean_w27 = _mean(total_w27)
    mean_w28 = _mean(total_w28)

    pct = lambda num, den: (100.0 * num / den) if den > 0 else 0.0

    # W28-specific stats.
    n_ratified = sum(1 for o in per_cell_producer_w28 if _w28_ratified(o))
    n_quorum_below = sum(
        1 for o in per_cell_producer_w28
        if _w28_branch(o) == W28_BRANCH_QUORUM_BELOW_THRESHOLD)
    n_passthrough = sum(
        1 for o in per_cell_producer_w28
        if _w28_branch(o) == W28_BRANCH_RATIFIED_PASSTHROUGH)
    n_no_trigger_w28 = sum(
        1 for o in per_cell_producer_w28
        if _w28_branch(o) == W28_BRANCH_NO_RATIFY_NEEDED)
    n_fallback_w28 = sum(
        1 for o in per_cell_producer_w28
        if _w28_branch(o) == W28_BRANCH_FALLBACK_W27)
    n_disabled_w28 = sum(
        1 for o in per_cell_producer_w28
        if _w28_branch(o) == W28_BRANCH_DISABLED)
    branch_counts_w28: dict[str, int] = {}
    for o in per_cell_producer_w28:
        b = _w28_branch(o)
        branch_counts_w28[b] = branch_counts_w28.get(b, 0) + 1

    overhead_per_cell = [
        max(0, w28_tokens_p[i] - w27_tokens_p[i])
        for i in range(n_cells_run)
    ]

    correctness_rate_w26 = (sum(correctness_w26) / n_cells_run
                              if n_cells_run else 0.0)
    correctness_rate_w27 = (sum(correctness_w27) / n_cells_run
                              if n_cells_run else 0.0)
    correctness_rate_w28 = (sum(correctness_w28) / n_cells_run
                              if n_cells_run else 0.0)

    # Ratification-aware correctness: a cell is "trusted-correct" iff
    # (ratified AND correct) OR (NOT ratified AND we abstained); the
    # bench cannot abstain on its answer (W26/W27 always emit), so
    # W28's contribution is to mark cells where we should NOT trust
    # the answer.
    n_ratified_correct = sum(
        1 for i in range(n_cells_run)
        if _w28_ratified(per_cell_producer_w28[i])
        and correctness_w28[i]
    )
    n_ratified_wrong = sum(
        1 for i in range(n_cells_run)
        if _w28_ratified(per_cell_producer_w28[i])
        and not correctness_w28[i]
    )
    n_unratified_correct = sum(
        1 for i in range(n_cells_run)
        if not _w28_ratified(per_cell_producer_w28[i])
        and correctness_w28[i]
    )
    n_unratified_wrong = sum(
        1 for i in range(n_cells_run)
        if not _w28_ratified(per_cell_producer_w28[i])
        and not correctness_w28[i]
    )

    # Trust precision = ratified_correct / (ratified_correct + ratified_wrong)
    trust_denom = n_ratified_correct + n_ratified_wrong
    trust_precision = (n_ratified_correct / trust_denom
                        if trust_denom > 0 else 0.0)
    trust_coverage = ((n_ratified_correct + n_ratified_wrong) / n_cells_run
                        if n_cells_run else 0.0)

    results: dict[str, Any] = {
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "n_cells": n_cells_run,
        "bank_seed": bank_seed,
        "chain_persist_window": chain_persist_window,
        "max_active_chains": effective_max_chains,
        "signature_period": signature_period,
        "registry_meta": registry_meta,

        # Visible-token comparisons (apples-to-apples per cell).
        "mean_total_w26_visible_tokens": round(mean_w26, 4),
        "mean_total_w27_visible_tokens": round(mean_w27, 4),
        "mean_total_w28_visible_tokens": round(mean_w28, 4),
        "mean_overhead_w28_vs_w27_per_cell": round(
            _mean(overhead_per_cell), 4),
        "max_overhead_w28_vs_w27_per_cell": int(
            max(overhead_per_cell) if overhead_per_cell else 0),

        # Equivalence at K=1 (H2 anchor).
        "byte_equivalent_w28_w27": (mean_w28 == mean_w27),

        # Correctness comparisons.
        "correctness_ratified_rate_w26": round(correctness_rate_w26, 4),
        "correctness_ratified_rate_w27": round(correctness_rate_w27, 4),
        "correctness_ratified_rate_w28": round(correctness_rate_w28, 4),
        # Variance proxies (per-cell correctness as 0/1).
        "correctness_stddev_w26": round(
            _stddev([int(c) for c in correctness_w26]), 4),
        "correctness_stddev_w27": round(
            _stddev([int(c) for c in correctness_w27]), 4),
        "correctness_stddev_w28": round(
            _stddev([int(c) for c in correctness_w28]), 4),

        # W28 ratification stats.
        "n_ratified": n_ratified,
        "n_quorum_below_threshold": n_quorum_below,
        "n_passthrough_w28_eq_w27": n_passthrough,
        "n_no_ratify_needed": n_no_trigger_w28,
        "n_fallback_w27": n_fallback_w28,
        "n_disabled_w28": n_disabled_w28,
        "branch_counts_w28": branch_counts_w28,

        # Trust diagnostics.
        "n_ratified_correct": n_ratified_correct,
        "n_ratified_wrong": n_ratified_wrong,
        "n_unratified_correct": n_unratified_correct,
        "n_unratified_wrong": n_unratified_wrong,
        "trust_precision": round(trust_precision, 4),
        "trust_coverage": round(trust_coverage, 4),

        # Probe / cross-host stats.
        "n_probe_calls_total": int(registry_w28.n_probe_calls_total),
        "n_cross_host_probe_calls": int(registry_w28.n_cross_host_probe_calls),
        "cross_host_round_trip_bytes":
            int(registry_w28.cross_host_round_trip_bytes),
        "n_ratifications_registered":
            int(registry_w28.n_ratifications_registered),
        "n_ratifications_rejected":
            int(registry_w28.n_ratifications_rejected),

        # Tampering bank.
        "n_tamper_attempts": n_tamper_attempts,
        "n_tampered_rejected": n_tampered_rejected,
    }

    if verbose:
        print(f"\n=== Phase 75 — R-75-{bank.upper()} ===")
        print(f"bank={bank}, T_decoder={T_decoder}, K={K_consumers}, "
              f"n_cells={n_cells_run}, "
              f"max_active_chains={effective_max_chains}")
        print(f"W26={mean_w26:.2f}  W27={mean_w27:.2f}  W28={mean_w28:.2f}  "
              f"overhead/cell={_mean(overhead_per_cell):.2f}")
        print(f"correctness W26={correctness_rate_w26:.4f} "
              f"W27={correctness_rate_w27:.4f} "
              f"W28={correctness_rate_w28:.4f}")
        print(f"ratified={n_ratified}/{n_cells_run}  "
              f"trust_precision={trust_precision:.4f}  "
              f"branches_w28={branch_counts_w28}")
        print(f"probe_calls={registry_w28.n_probe_calls_total}  "
              f"cross_host_calls={registry_w28.n_cross_host_probe_calls}  "
              f"cross_host_bytes={registry_w28.cross_host_round_trip_bytes}")
        if bank == "ratification_tampered":
            print(f"tampered_rejected={n_tampered_rejected}/"
                  f"{n_tamper_attempts}")

    return results


def run_phase75_seed_stability_sweep(
        *,
        seeds: tuple[int, ...] = (11, 17, 23, 29, 31),
        bank: str = "cross_model_drift",
        T_decoder: int | None = None,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        chain_persist_window: int | None = None,
        max_active_chains: int = 8,
        signature_period: int = 4,
        verbose: bool = False,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        r = run_phase75(
            bank=bank, T_decoder=T_decoder, K_consumers=K_consumers,
            n_eval=n_eval, bank_replicates=bank_replicates,
            bank_seed=seed,
            chain_persist_window=chain_persist_window,
            max_active_chains=max_active_chains,
            signature_period=signature_period,
            verbose=verbose)
        rows.append(r)
    overhead = [r["mean_overhead_w28_vs_w27_per_cell"] for r in rows]
    correctness_w28 = [r["correctness_ratified_rate_w28"] for r in rows]
    correctness_w27 = [r["correctness_ratified_rate_w27"] for r in rows]
    trust_prec = [r["trust_precision"] for r in rows]
    cross_host_bytes = [r["cross_host_round_trip_bytes"] for r in rows]
    return {
        "seeds": list(seeds),
        "bank": bank,
        "T_decoder": T_decoder,
        "K_consumers": K_consumers,
        "max_active_chains": max_active_chains,
        "rows": rows,
        "min_overhead_w28_vs_w27": min(overhead),
        "mean_overhead_w28_vs_w27": sum(overhead) / len(overhead),
        "max_overhead_w28_vs_w27": max(overhead),
        "min_correctness_w28": min(correctness_w28),
        "min_correctness_w27": min(correctness_w27),
        "all_correctness_w28_ge_w27": all(
            cw28 >= cw27 for cw28, cw27 in zip(correctness_w28, correctness_w27)
        ),
        "min_trust_precision": min(trust_prec),
        "mean_trust_precision": sum(trust_prec) / len(trust_prec),
        "any_cross_host_probing": any(b > 0 for b in cross_host_bytes),
        "max_cross_host_round_trip_bytes": max(cross_host_bytes)
            if cross_host_bytes else 0,
    }


def run_cross_regime_p75(
        *,
        K_consumers: int = 3,
        n_eval: int = 16,
        bank_replicates: int = 4,
        bank_seed: int = 11,
        verbose: bool = False,
) -> dict[str, Any]:
    """Run the full R-75 family (synthetic; cross_host_live optional)."""
    sub_banks = (
        "single_probe", "chain_shared",
        "cross_model_drift", "coordinated_drift", "trust_zero",
        "ratification_tampered", "pool_exhausted",
    )
    out = {}
    for b in sub_banks:
        out[b] = run_phase75(
            bank=b, T_decoder=None,
            K_consumers=K_consumers, n_eval=n_eval,
            bank_replicates=bank_replicates,
            bank_seed=bank_seed, verbose=verbose)
    return out


def _cli() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 75 — W28 ensemble-verified multi-chain pivot ratification")
    ap.add_argument("--bank", default="cross_model_drift",
                     choices=["single_probe", "chain_shared",
                                "cross_model_drift", "coordinated_drift",
                                "trust_zero", "ratification_tampered",
                                "pool_exhausted", "cross_host_live",
                                "cross_regime", "topology_probe"])
    ap.add_argument("--K-consumers", type=int, default=3)
    ap.add_argument("--decoder-budget", type=int, default=-1)
    ap.add_argument("--chain-persist-window", type=int, default=-1)
    ap.add_argument("--max-active-chains", type=int, default=8)
    ap.add_argument("--signature-period", type=int, default=4)
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--bank-replicates", type=int, default=4)
    ap.add_argument("--bank-seed", type=int, default=11)
    ap.add_argument("--seed-sweep", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", default="-")
    args = ap.parse_args()

    T_decoder: int | None = (None if args.decoder_budget < 0
                              else args.decoder_budget)
    chain_persist_window: int | None = (
        None if args.chain_persist_window < 0
        else args.chain_persist_window)

    if args.bank == "cross_regime":
        result = run_cross_regime_p75(
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed, verbose=args.verbose)
    elif args.bank == "topology_probe":
        result = discover_two_host_topology()
    elif args.seed_sweep:
        result = run_phase75_seed_stability_sweep(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            chain_persist_window=chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            verbose=args.verbose)
    else:
        result = run_phase75(
            bank=args.bank, T_decoder=T_decoder,
            K_consumers=args.K_consumers, n_eval=args.n_eval,
            bank_replicates=args.bank_replicates,
            bank_seed=args.bank_seed,
            chain_persist_window=chain_persist_window,
            max_active_chains=args.max_active_chains,
            signature_period=args.signature_period,
            verbose=args.verbose)

    out_text = json.dumps(result, indent=2, default=str)
    if args.out == "-":
        print(out_text)
    else:
        with open(args.out, "w") as fh:
            fh.write(out_text)


if __name__ == "__main__":
    _cli()
