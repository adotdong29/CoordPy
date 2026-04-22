"""Collaborative-build task — real interconnected work for hundreds of agents.

Goal: jointly produce a **product specification** for a fictional software
product ("PulseCore Dashboard v2"). The spec decomposes into ~40 subtasks
across ~10 specialties:

    product                 — overall goals, user personas
    data_model              — entities, relationships, schema
    api_design              — endpoints, request/response shapes
    auth                    — authn/authz model
    frontend_routing        — page structure
    frontend_components     — individual UI widgets
    backend_services        — microservice boundaries
    data_pipeline           — ETL, streaming
    observability           — metrics, logs, traces
    testing                 — unit, integration, e2e strategy
    deployment              — CI/CD, infra, rollout
    security                — threat model, hardening
    docs                    — user / API / internal

Each subtask has explicit **dependencies** on upstream subtasks, forming
a DAG. Agents self-claim subtasks matching their specialty key. The
network routes completion messages to downstream agents via MoE.

Scoring:
  - Completion: % of the 40 subtasks finished
  - Integration quality: does the final artifact reference each upstream
    output (cross-coverage)?
  - Inter-agent message count (shouldn't explode with team size)

This is the "team actually building together" test — different agents
do different things, and the outputs must fit together.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from vision_mvp.core.task_board import Subtask
from vision_mvp.core.agent_keys import l2_normalize


SPECIALTIES = [
    "product", "data_model", "api_design", "auth",
    "frontend_routing", "frontend_components",
    "backend_services", "data_pipeline",
    "observability", "testing", "deployment",
    "security", "docs",
]


SPECIALTY_DESCRIPTIONS = {
    "product": "Product manager: user personas, JTBD, success metrics, feature scope.",
    "data_model": "Data architect: entities, relationships, schema, validation rules.",
    "api_design": "API designer: REST/GraphQL endpoints, request/response shapes, versioning.",
    "auth": "Auth engineer: authentication (JWT/OAuth), authorization (RBAC), sessions.",
    "frontend_routing": "Frontend routing engineer: page structure, nav, URL design.",
    "frontend_components": "Frontend engineer: individual React/Vue components, forms, tables.",
    "backend_services": "Backend architect: microservice decomposition, bounded contexts.",
    "data_pipeline": "Data engineer: ETL jobs, streaming, data contracts.",
    "observability": "SRE: metrics (Prometheus), logs, traces, alerting rules.",
    "testing": "Test engineer: unit/integration/e2e coverage, test pyramid, fixtures.",
    "deployment": "Platform engineer: CI/CD pipeline, infra-as-code, rollout strategy.",
    "security": "Security engineer: threat model, OWASP top-10, secrets management.",
    "docs": "Tech writer: user docs, API reference, runbooks.",
}


def _spec_subtasks() -> list[tuple[str, str, str, list[str], str]]:
    """Return the fixed list of (id, specialty, title, deps, description)."""
    # Each row: (id, specialty, title, deps list, description)
    return [
        # -------- Foundations (no deps) --------
        ("ST01", "product", "Define primary user personas", [],
         "Identify the 2-3 primary user personas for PulseCore Dashboard v2."),
        ("ST02", "product", "List top-5 user JTBD", [],
         "List the top-5 jobs-to-be-done the dashboard must satisfy."),
        ("ST03", "product", "Success metrics", [],
         "Define product success metrics (activation, engagement, retention)."),
        ("ST04", "security", "Threat model", [],
         "Draft a STRIDE threat model — top 5 threats and mitigations."),

        # -------- Depends on product --------
        ("ST05", "data_model", "Core entities + schema", ["ST01", "ST02"],
         "Based on personas and JTBD, list core entities and their fields."),
        ("ST06", "data_model", "Relationships + constraints", ["ST05"],
         "Entity relationships and referential integrity constraints."),
        ("ST07", "api_design", "Top 10 API endpoints", ["ST05", "ST06"],
         "Propose the 10 most important REST endpoints with methods and paths."),
        ("ST08", "api_design", "Versioning + error model", ["ST07"],
         "API versioning strategy and standard error-response envelope."),

        # -------- Auth + sessions --------
        ("ST09", "auth", "Authentication method", ["ST01"],
         "Choose authN method (OAuth 2.1? SAML? passkeys?) and justify."),
        ("ST10", "auth", "Authorization (RBAC) model", ["ST05", "ST09"],
         "Role hierarchy and permission matrix."),

        # -------- Frontend --------
        ("ST11", "frontend_routing", "Page inventory + URL map", ["ST01", "ST02"],
         "List every top-level page with its URL and primary purpose."),
        ("ST12", "frontend_components", "Key components (8)", ["ST11"],
         "List the 8 most-reused React components and their props."),
        ("ST13", "frontend_components", "Form-validation patterns", ["ST07", "ST12"],
         "Shared validation patterns and error UX."),

        # -------- Backend services --------
        ("ST14", "backend_services", "Service boundaries", ["ST05", "ST07"],
         "Propose 3-5 backend services (bounded contexts)."),
        ("ST15", "backend_services", "Synchronous vs event flows", ["ST14"],
         "For each service-to-service interaction, pick sync HTTP or async events."),
        ("ST16", "backend_services", "Data ownership matrix", ["ST05", "ST14"],
         "Which service owns which entities (no shared-writer)."),

        # -------- Data pipeline --------
        ("ST17", "data_pipeline", "Top 5 ETL jobs", ["ST05"],
         "The 5 most important batch/streaming data jobs."),
        ("ST18", "data_pipeline", "Data contracts between services", ["ST15", "ST17"],
         "Schema contracts for cross-service events."),

        # -------- Observability --------
        ("ST19", "observability", "Service-level indicators", ["ST03", "ST14"],
         "SLIs per service (latency, error rate, saturation)."),
        ("ST20", "observability", "Alert catalog", ["ST19"],
         "10 most important alerts with thresholds."),
        ("ST21", "observability", "Log + trace schema", ["ST14", "ST18"],
         "Standard log fields and distributed-trace propagation."),

        # -------- Testing --------
        ("ST22", "testing", "Unit-test strategy", ["ST14"],
         "Per-service unit testing: frameworks, coverage targets."),
        ("ST23", "testing", "Integration-test scope", ["ST15", "ST14"],
         "Which service-pairs need integration tests and why."),
        ("ST24", "testing", "E2E top-10 flows", ["ST11", "ST12", "ST07"],
         "The 10 user journeys to cover end-to-end."),

        # -------- Deployment --------
        ("ST25", "deployment", "CI/CD pipeline stages", ["ST14"],
         "Pipeline from commit to prod: stages, gates, approvals."),
        ("ST26", "deployment", "Infra-as-code approach", ["ST14"],
         "Terraform or Pulumi? Modules? Environments?"),
        ("ST27", "deployment", "Rollout strategy", ["ST19", "ST20", "ST25"],
         "Canary / blue-green / feature flags."),

        # -------- Security hardening --------
        ("ST28", "security", "Secrets management", ["ST25", "ST09"],
         "How secrets are stored and rotated in CI/CD + runtime."),
        ("ST29", "security", "Input validation strategy", ["ST07", "ST13"],
         "Shared input-validation approach across API and frontend."),

        # -------- Docs --------
        ("ST30", "docs", "User guide outline", ["ST11", "ST24"],
         "Outline the user guide table of contents."),
        ("ST31", "docs", "API reference layout", ["ST07", "ST08"],
         "How API docs are structured and where they live."),
        ("ST32", "docs", "Runbook catalog", ["ST19", "ST20", "ST27"],
         "Runbooks for the top-10 SRE alerts."),

        # -------- Integration / synthesis --------
        ("ST33", "product", "Rollout + launch plan", ["ST27", "ST32"],
         "Phased launch with milestones tied to observability readiness."),
        ("ST34", "product", "Feature flag inventory", ["ST27", "ST29"],
         "Initial feature flags for rollout."),
        ("ST35", "observability", "Metrics dashboard spec", ["ST19", "ST20"],
         "Key Grafana dashboard layout."),

        # -------- Cross-cutting --------
        ("ST36", "security", "Privacy + data retention", ["ST05", "ST17"],
         "PII identification, retention periods, deletion flow."),
        ("ST37", "docs", "Internal architecture doc", ["ST14", "ST18", "ST21"],
         "Single-page internal overview."),
        ("ST38", "testing", "Chaos-engineering scope", ["ST19", "ST27"],
         "Chaos experiments + failure domains."),
        ("ST39", "deployment", "Disaster-recovery plan", ["ST19", "ST26"],
         "RTO, RPO, and step-by-step recovery."),
        ("ST40", "product", "Executive summary", ["ST33", "ST37"],
         "Final 1-page exec summary synthesizing the spec."),
    ]


def make_subtasks(embed_fn) -> list[Subtask]:
    """Build the full subtask list with tag_embeddings computed via embed_fn."""
    out = []
    for (sid, specialty, title, deps, desc) in _spec_subtasks():
        # Tag embedding = embedding of "specialty: title"
        tag_text = f"{specialty}: {title}. {desc}"
        tag_emb = l2_normalize(np.asarray(embed_fn(tag_text), dtype=np.float64))
        out.append(Subtask(
            id=sid, title=title, description=desc,
            tag_embedding=tag_emb, deps=list(deps),
        ))
    return out


def assign_agent_specialties(n_agents: int, seed: int = 0) -> list[str]:
    """Distribute agents across specialties in a roughly-even way."""
    rng = np.random.default_rng(seed)
    base = list(SPECIALTIES)
    # Repeat then shuffle
    per = max(1, n_agents // len(base) + 1)
    roles = base * per
    rng.shuffle(roles)
    return roles[:n_agents]


def score_build(board) -> dict:
    """Score the collaborative build."""
    n_done = board.done_count()
    n_total = board.total_count()
    completion = n_done / max(n_total, 1)

    # Integration: each done task that has deps should reference content from
    # at least half its deps' outputs. Heuristic: check for shared tokens.
    integration_score = 0.0
    integrations_checked = 0
    for t in board.subtasks.values():
        if t.status != "done" or not t.deps:
            continue
        integrations_checked += 1
        own = set(t.output.lower().split())
        hits = 0
        for dep_id in t.deps:
            dep = board.subtasks.get(dep_id)
            if dep is None or dep.status != "done" or not dep.output:
                continue
            dep_tokens = set(dep.output.lower().split())
            # any 3+ dep-output token appearing in own output?
            long_tokens = {tok for tok in dep_tokens if len(tok) >= 5}
            overlap = own & long_tokens
            if len(overlap) >= 2:
                hits += 1
        if t.deps:
            integration_score += hits / len(t.deps)
    if integrations_checked > 0:
        integration_score /= integrations_checked

    return {
        "completion_rate": round(completion, 3),
        "n_done": n_done,
        "n_total": n_total,
        "integration_score": round(integration_score, 3),
        "integrations_checked": integrations_checked,
    }
