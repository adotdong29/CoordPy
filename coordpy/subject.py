"""coordpy.subject — the executable orientation surface for CoordPy.

This module answers, in **one deterministic run**, the questions a new
operator or agent has to answer before doing anything else:

  * **What is CoordPy?**  A Python-first agent development kit (ADK):
    Agents, Tools, Sessions, and a Runner (``coordpy.adk``), with a
    context-capsule runtime underneath — every piece of context that
    crosses a role / layer / run boundary is a typed, content-addressed,
    lifecycle-bounded, provenance-carrying ``ContextCapsule`` (the C1..C6
    contract) you can re-verify and replay.
  * **What is the main tool?**  ``coordpy.adk`` (import-and-code) for
    *building* with CoordPy; ``coordpy-team`` for the secondary CLI; and
    ``coordpy-subject`` (this surface) for *understanding / verifying* it.
  * **What is stable vs experimental vs historical?**  The S1..S5 tier
    map below, mirrored in ``docs/W144_COORDPY_SUBJECT_REGISTRY.json``.
  * **Does the stable contract still hold here?**  Run the hermetic
    harness (``coordpy-subject check``): stable SDK smoke, capsule
    verification, team runtime, and curated experimental-exemplar
    imports — no network, no model, deterministic.

Context Zero is the *research programme*; CoordPy is the *shipped
product* produced by it. This surface describes the product and points
at the programme; it does not absorb the benchmark arc into the product.

Stability: ``coordpy.subject`` is reachable via explicit import and the
``coordpy-subject`` console script / ``python -m coordpy.subject``. It is
additive — it does not change the released stable public surface.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

SUBJECT_REPORT_SCHEMA = "coordpy.subject.v1"

# --------------------------------------------------------------------------
# Canonical subject statement (mechanism-shaped, per master plan §10.0).
# --------------------------------------------------------------------------

CANONICAL_SUBJECT = {
    "name": "CoordPy",
    "one_line": (
        "CoordPy is a Python-first agent development kit (ADK): you build "
        "Agents, Tools, Sessions, and sub-agents and run them with a Runner, "
        "while every model call, tool call, and handoff automatically seals "
        "into a typed, content-addressed, lifecycle-bounded, provenance-"
        "carrying capsule you can re-verify and replay."
    ),
    "centre_of_gravity": (
        "The ADK ergonomics are the front door (coordpy.adk); the Capsule "
        "Contract (C1..C6) is the guarantee underneath."
    ),
    "capsule_contract": {
        "C1": "Identity — SHA-256 content-address over (kind, payload, budget, parents).",
        "C2": "Typed claim — CapsuleKind in a closed vocabulary.",
        "C3": "Lifecycle — PROPOSED -> ADMITTED -> SEALED (+ optional RETIRED); illegal transitions refused.",
        "C4": "Budget — CapsuleBudget enforced at admit time.",
        "C5": "Provenance — parents must be in the ledger; the hash chain detects retroactive inserts.",
        "C6": "Frozen — a sealed capsule's CID is fixed for all time.",
    },
    "programme_product_split": {
        "Context Zero": "the research programme (the W1..W144 arc).",
        "CoordPy": "the shipped product — a Python-first ADK (library + secondary CLI). Not the whole programme; not a universal agent platform.",
    },
    "operational_surfaces": [
        "library ADK surface (coordpy.adk: Agent / Tool / Runner / Session / State / Memory / Artifacts)",
        "team runtime (AgentTeam / coordpy-team)",
        "provenance + capsule audit (coordpy-capsule)",
        "reproducible profile runs (coordpy / RunSpec->RunReport)",
        "CI gate (coordpy-ci) and import audit (coordpy-import)",
    ],
}

FRONT_DOORS = {
    "library": {
        "entrypoint": "coordpy.adk",
        "why": "Import-and-code ADK surface (Agent / Tool / Runner / Session / State / Memory / Artifacts). The recommended door for BUILDING with CoordPy.",
    },
    "usage": {
        "entrypoint": "coordpy-team",
        "why": "Run / replay / sweep / compare an AgentTeam preset from the CLI. The secondary command-line runtime surface.",
    },
    "orientation": {
        "entrypoint": "coordpy-subject",
        "why": "Print the subject, the S1..S5 tier map, and run the hermetic harness. The door for UNDERSTANDING / VERIFYING CoordPy.",
    },
    "secondary": ["coordpy-team (CLI)", "coordpy (profile runner)", "coordpy-capsule (audit)", "coordpy-import", "coordpy-ci"],
}

# --------------------------------------------------------------------------
# S1..S5 tier map (the canonical classification, mirrored in the registry
# doc). S1/S2 are enumerated explicitly; S3/S4 are pattern + representative
# members; S5 is the named keep-for-record set.
# --------------------------------------------------------------------------

TIERS: dict[str, dict[str, Any]] = {
    "S1_stable_core": {
        "definition": (
            "On the released SDK/CLI public contract: a stable "
            "(non-__experimental__) symbol re-exported by coordpy/__init__.py, "
            "or one of the console scripts / on-disk schemas / _internal.product "
            "modules / extensions. Contract-test-locked."
        ),
        "modules": [
            "config", "provenance", "run", "runtime", "llm_backend",
            "capsule", "capsule_runtime", "lifecycle_audit", "api_layers",
            "agents", "presets", "synthetic_llm",
            "capsule_policy", "capsule_policy_bundle",
            "capsule_decoder", "capsule_decoder_v2",
            "team_coord", "team_policy", "extensions",
            "_cli", "_version", "subject", "adk",
            "_internal.product.profiles", "_internal.product.report",
            "_internal.product.ci_gate", "_internal.product.import_data",
        ],
        "console_scripts": [
            "coordpy-team (USAGE front door)", "coordpy-subject (ORIENTATION front door)",
            "coordpy-capsule", "coordpy", "coordpy-import", "coordpy-ci",
        ],
        "schemas": [
            "coordpy.capsule_view.v1", "coordpy.team_result.v1",
            "coordpy.provenance.v1", "phase45.product_report.v2",
            "phase46.ci_verdict.v1", "phase46.import_audit.v1",
            "coordpy.subject.v1",
        ],
        "note": (
            "team_coord additionally HOSTS __experimental__ W10..W40 decoder "
            "symbols; its stable core (TeamCoordinator, capsule_team_handoff/"
            "role_view/team_decision, audit_team_lifecycle) is the S1 team layer."
        ),
    },
    "S2_canonical_experimental": {
        "definition": (
            "Explicit-import-only, but central to CoordPy's identity: on the "
            "architecture north-star lineage OR named load-bearing in the current "
            "RESEARCH_STATUS top markers (W140..W143). Ships in the wheel, flagged "
            "unstable, reachable only via explicit import."
        ),
        "architecture_north_star_lineage": [
            "product_manifold", "live_manifold", "learned_manifold",
            "autograd_manifold", "shared_state_proxy",
        ],
        "discover_then_amortize_chain_W140_W143": [
            "family_tutor_compiler_v1", "self_tutoring_controller_v1",
            "self_tutoring_technique_extractor_v1",
            "no_oracle_verifier_v1", "no_oracle_verifier_v2",
            "multi_agent_discover_amortize_v1",
        ],
        "note": (
            "North-star name: 'Latent State Transition' (the next-branch "
            "architecture target, NOT the shipped product). The W140..W143 chain "
            "is the most recent load-bearing research result."
        ),
    },
    "S3_benchmark_research_support": {
        "definition": (
            "Evaluation machinery: load-bearing to RESULTS, not to the product "
            "identity. Must stop presenting as central."
        ),
        "patterns": [
            "*_bench_v*", "*_corpus_v*", "*_battlefield_v*", "*_slate_v*",
            "*_loader_v*", "*_executor_v*", "*_preflight_v*",
            "realworldqa_*", "icpc_*", "apps_*", "bigcodebench_*", "mbpp_*",
        ],
        "examples": [
            "resistant_by_construction_battlefield_v1", "hard_battlefield_slate_v2",
            "realworldqa_bench_v2", "icpc_reflexion_bench_v1",
            "moderate_p_family_screen_v1",
        ],
    },
    "S4_historical_archive": {
        "definition": (
            "Kept verbatim for provenance. Explicit-import-only; never on the "
            "product front door."
        ),
        "families": {
            "W22..W42 dense-control / trust-adjudication ladder": "188 __experimental__ symbols (incl. integrated_synthesis [W41], role_invariant_synthesis [W42], eagerly imported for back-compat but flagged __experimental__).",
            "W49..W139 substrate version ladders": "tiny_substrate_v*, persistent_latent_v* (28), kv_bridge_v* (24), long_horizon_retention_v* (29), mergeable_latent_capsule_v* (27), deep_substrate_hybrid_v*, consensus_fallback_controller_v*, etc. (~400+ modules).",
            "per-milestone composition modules": "w*_team, r*_benchmark.",
        },
    },
    "S5_blocked_dead_keep_for_record": {
        "definition": "Explicitly killed or superseded; KEPT (provenance), not deleted.",
        "members": [
            "hosted_cost_planner_v10 (superseded by v11/v12)",
            "hosted_logprob_router_v10 (superseded by v11/v12)",
            "differentiable_memory / composed_learned_memory / live_composed_* "
            "(W136-killed: random-until-trained nets / GPU-blocked, where present)",
        ],
        "note": "Census found only 4 true orphans repo-wide (__main__ and _pretty are legitimate internals). Nothing dead is deleted in W144.",
    },
}

# Curated S2 exemplars the harness import-checks (module -> a documented symbol).
_S2_EXEMPLARS: tuple[tuple[str, str], ...] = (
    ("coordpy.product_manifold", "ProductManifoldChannelBundle"),
    ("coordpy.learned_manifold", "LearnedManifoldOrchestrator"),
    ("coordpy.shared_state_proxy", "SharedStateCapsule"),
    ("coordpy.family_tutor_compiler_v1", "FamilyTutorV1"),
    ("coordpy.self_tutoring_controller_v1", "discover_self_scaffold_v1"),
    ("coordpy.no_oracle_verifier_v2", "select_winner_v2"),
    ("coordpy.multi_agent_discover_amortize_v1", "team_discover_v1"),
)

# The stable public symbols the smoke check asserts are present on ``coordpy``.
_STABLE_PUBLIC_SYMBOLS: tuple[str, ...] = (
    "RunSpec", "run", "RunReport", "SweepSpec", "run_sweep", "CoordPyConfig",
    "Agent", "AgentTurn", "ActionDecision", "AgentTeam", "TeamResult",
    "agent", "create_team", "replay_team_result", "presets",
    "TEAM_RESULT_SCHEMA", "profiles", "report", "ci_gate", "import_data",
    "extensions", "ContextCapsule", "CapsuleLedger", "CapsuleView",
    "verify_chain_from_view_dict", "render_view", "CAPSULE_VIEW_SCHEMA",
    "OpenAICompatibleBackend", "OllamaBackend", "backend_from_env",
    "PROVENANCE_SCHEMA", "build_manifest",
    "__version__", "SDK_VERSION", "PRODUCT_REPORT_SCHEMA",
    "adk",  # W145 — the library-first ADK front door
)

EXPECTED_VERSION = "0.5.20"
EXPECTED_SDK_VERSION = "coordpy.sdk.v3.43"


# --------------------------------------------------------------------------
# Harness checks. Each returns {name, status: PASS|FAIL, detail}.
# All hermetic: no network, no real model.
# --------------------------------------------------------------------------

def _ok(name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "PASS", "detail": detail}


def _fail(name: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": "FAIL", "detail": detail}


def check_stable_smoke() -> dict[str, str]:
    """Import coordpy, assert version invariants + stable public surface."""
    try:
        import coordpy
        if coordpy.__version__ != EXPECTED_VERSION:
            return _fail("stable_smoke",
                         f"__version__ {coordpy.__version__!r} != {EXPECTED_VERSION!r}")
        if coordpy.SDK_VERSION != EXPECTED_SDK_VERSION:
            return _fail("stable_smoke",
                         f"SDK_VERSION {coordpy.SDK_VERSION!r} != {EXPECTED_SDK_VERSION!r}")
        missing = [s for s in _STABLE_PUBLIC_SYMBOLS if not hasattr(coordpy, s)]
        if missing:
            return _fail("stable_smoke", f"missing stable symbols: {missing}")
        return _ok("stable_smoke",
                   f"coordpy {coordpy.__version__} ({coordpy.SDK_VERSION}); "
                   f"{len(_STABLE_PUBLIC_SYMBOLS)} stable symbols present; "
                   f"{len(coordpy.__experimental__)} __experimental__ symbols flagged")
    except Exception as exc:  # pragma: no cover - defensive
        return _fail("stable_smoke", f"{type(exc).__name__}: {exc}")


def check_team_runtime() -> tuple[dict[str, str], dict[str, Any] | None]:
    """Run a hermetic 2-agent team via SyntheticLLMClient; return its view."""
    try:
        from coordpy import create_team, agent
        from coordpy.synthetic_llm import SyntheticLLMClient
        team = create_team(
            [agent("planner", "Break the task into 2 steps."),
             agent("writer", "Write the final answer from the prior handoffs.")],
            backend=SyntheticLLMClient(default_response="ok: bounded-context handoff"),
            team_instructions="Reuse visible handoffs instead of restating the task.",
            max_visible_handoffs=3,
        )
        result = team.run("Explain what CoordPy does in one sentence.")
        if not result.final_output:
            return _fail("team_runtime", "empty final_output"), None
        view = result.capsule_view
        if not view or view.get("schema") != "coordpy.capsule_view.v1":
            return _fail("team_runtime",
                         f"missing/bad capsule_view schema: {view and view.get('schema')}"), view
        cram = result.cramming_estimate()
        return _ok("team_runtime",
                   f"{len(result.turns)} turns; capsule_view {view['schema']} "
                   f"chain_head={str(view.get('chain_head'))[:12]}; "
                   f"savings={cram.get('savings_pct', 0.0):.0f}%"), view
    except Exception as exc:
        return _fail("team_runtime", f"{type(exc).__name__}: {exc}"), None


def check_capsule_verify(view: dict[str, Any] | None) -> dict[str, str]:
    """Re-hash the team run's sealed capsule chain from the view dict alone."""
    try:
        from coordpy import verify_chain_from_view_dict
        if view is None:
            return _fail("capsule_verify", "no capsule view available from team run")
        ok = verify_chain_from_view_dict(view)
        if not ok:
            return _fail("capsule_verify", "verify_chain_from_view_dict returned False")
        return _ok("capsule_verify",
                   f"chain re-verified from view bytes; chain_ok={view.get('chain_ok')}; "
                   f"n_capsules={len(view.get('capsules', []))}")
    except Exception as exc:
        return _fail("capsule_verify", f"{type(exc).__name__}: {exc}")


def check_s2_exemplars() -> dict[str, str]:
    """Import each curated canonical-experimental exemplar + a documented symbol."""
    missing = []
    for mod_name, symbol in _S2_EXEMPLARS:
        try:
            mod = importlib.import_module(mod_name)
            if not hasattr(mod, symbol):
                missing.append(f"{mod_name}.{symbol} (symbol absent)")
        except Exception as exc:
            missing.append(f"{mod_name} ({type(exc).__name__}: {exc})")
    if missing:
        return _fail("s2_exemplars", "; ".join(missing))
    return _ok("s2_exemplars",
               f"all {len(_S2_EXEMPLARS)} canonical-experimental exemplars import "
               f"with documented symbols present")


def run_harness() -> list[dict[str, str]]:
    """Run every hermetic check; return the ordered check list."""
    checks: list[dict[str, str]] = []
    checks.append(check_stable_smoke())
    team_check, view = check_team_runtime()
    checks.append(team_check)
    checks.append(check_capsule_verify(view))
    checks.append(check_s2_exemplars())
    return checks


# --------------------------------------------------------------------------
# Report assembly + rendering.
# --------------------------------------------------------------------------

def build_subject_report(run_checks: bool = True) -> dict[str, Any]:
    """Assemble the deterministic ``coordpy.subject.v1`` report.

    Set ``run_checks=False`` for a pure subject/registry description with
    no harness execution.
    """
    report: dict[str, Any] = {
        "schema": SUBJECT_REPORT_SCHEMA,
        "subject": CANONICAL_SUBJECT,
        "front_doors": FRONT_DOORS,
        "tiers": TIERS,
        "out_of_scope": [
            "the Context Zero benchmark arc (W89..W143 results) — S3, not the product",
            "the 'Latent State Transition' architecture branch — next-programme, not shipped",
            "transformer-internal trust transfer / hidden-state access",
        ],
        "registry_doc": "docs/W144_COORDPY_SUBJECT_REGISTRY.json",
    }
    if run_checks:
        checks = run_harness()
        report["checks"] = checks
        report["all_pass"] = all(c["status"] == "PASS" for c in checks)
    return report


def render_text(report: dict[str, Any]) -> str:
    s = report["subject"]
    lines: list[str] = []
    lines.append("CoordPy — subject harness (coordpy.subject.v1)")
    lines.append("=" * 52)
    lines.append("")
    lines.append("WHAT IT IS")
    lines.append("  " + s["one_line"])
    lines.append(f"  Centre of gravity: {s['centre_of_gravity']}")
    lines.append("")
    lines.append("FRONT DOORS")
    fd = report["front_doors"]
    if "library" in fd:
        lines.append(f"  BUILD      -> {fd['library']['entrypoint']}: {fd['library']['why']}")
    lines.append(f"  USE (CLI)  -> {fd['usage']['entrypoint']}: {fd['usage']['why']}")
    lines.append(f"  UNDERSTAND -> {fd['orientation']['entrypoint']}: {fd['orientation']['why']}")
    lines.append(f"  secondary  -> {', '.join(fd['secondary'])}")
    lines.append("")
    lines.append("TIERS (stable -> historical)")
    t = report["tiers"]
    lines.append(f"  S1 stable-core: {len(t['S1_stable_core']['modules'])} modules, "
                 f"{len(t['S1_stable_core']['console_scripts'])} console scripts, "
                 f"{len(t['S1_stable_core']['schemas'])} on-disk schemas")
    s2 = t["S2_canonical_experimental"]
    n_s2 = len(s2["architecture_north_star_lineage"]) + len(s2["discover_then_amortize_chain_W140_W143"])
    lines.append(f"  S2 canonical-experimental: {n_s2} modules "
                 "(manifold north-star lineage + W140..W143 discover-then-amortize)")
    lines.append("  S3 benchmark/research-support: the eval machinery (bench/corpus/battlefield/loaders)")
    lines.append("  S4 historical/archive: W22..W42 dense-control + W49..W139 substrate version ladders (~400+)")
    lines.append("  S5 blocked/dead/keep-for-record: superseded version tails + W136-killed memory trio (kept)")
    lines.append("")
    lines.append("OUT OF SCOPE (not the product)")
    for item in report["out_of_scope"]:
        lines.append(f"  - {item}")
    if "checks" in report:
        lines.append("")
        lines.append("HARNESS (hermetic; no network/model)")
        for c in report["checks"]:
            mark = "PASS" if c["status"] == "PASS" else "FAIL"
            lines.append(f"  [{mark}] {c['name']}: {c['detail']}")
        lines.append("")
        lines.append(f"  OVERALL: {'ALL PASS' if report['all_pass'] else 'FAILURES PRESENT'}")
    lines.append("")
    lines.append(f"Registry: {report['registry_doc']}  |  "
                 "Programme=Context Zero, Product=CoordPy")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if "--version" in argv:
        import coordpy
        print(f"coordpy {coordpy.__version__} ({coordpy.SDK_VERSION})")
        return 0
    ap = argparse.ArgumentParser(
        prog="coordpy-subject",
        description="CoordPy orientation surface: what CoordPy is, the "
                    "stable/experimental/historical tier map, and a hermetic "
                    "harness over the stable contract.")
    ap.add_argument("--version", action="store_true",
                    help="print CoordPy SDK version and exit")
    sub = ap.add_subparsers(dest="sub")
    p_report = sub.add_parser("report", help="full subject report + harness (default)")
    p_report.add_argument("--json", action="store_true", help="emit JSON")
    p_report.add_argument("--no-checks", action="store_true",
                          help="describe the subject without running the harness")
    p_check = sub.add_parser("check", help="run the hermetic harness; exit non-zero on any failure")
    p_check.add_argument("--json", action="store_true", help="emit JSON")
    p_reg = sub.add_parser("registry", help="emit the S1..S5 machine-readable registry as JSON")
    p_tiers = sub.add_parser("tiers", help="print the S1..S5 tier map (text)")
    args = ap.parse_args(argv)

    sub_cmd = args.sub or "report"

    if sub_cmd == "registry":
        print(json.dumps({"schema": SUBJECT_REPORT_SCHEMA,
                          "subject": CANONICAL_SUBJECT,
                          "front_doors": FRONT_DOORS,
                          "tiers": TIERS}, indent=2, sort_keys=True))
        return 0

    if sub_cmd == "tiers":
        print(render_text(build_subject_report(run_checks=False)))
        return 0

    if sub_cmd == "check":
        checks = run_harness()
        all_pass = all(c["status"] == "PASS" for c in checks)
        if getattr(args, "json", False):
            print(json.dumps({"schema": SUBJECT_REPORT_SCHEMA,
                              "checks": checks, "all_pass": all_pass},
                             indent=2, sort_keys=True))
        else:
            for c in checks:
                print(f"[{c['status']}] {c['name']}: {c['detail']}")
            print(f"OVERALL: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")
        return 0 if all_pass else 1

    # default: report
    report = build_subject_report(run_checks=not getattr(args, "no_checks", False))
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    # A report run exits non-zero if the harness ran and any check failed.
    if "all_pass" in report and not report["all_pass"]:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
