"""W144 unit tests — the executable CoordPy subject harness (Lane β).

Fast $0 tests (no NIM, no network): the ``coordpy.subject`` orientation
surface must (1) preserve the stable boundary (0.5.20 / coordpy.sdk.v3.43),
(2) run a hermetic harness whose 4 checks all PASS (stable smoke, team
runtime, capsule re-verify, S2 exemplar imports), (3) expose a complete
S1..S5 tier map whose S1 list matches the documented stable surface,
(4) match the on-disk machine-readable registry doc, and (5) the CLI
exit-code contract holds. It also closes the coverage gap W144 surfaced:
the three previously-untested S2 discover-then-amortize chain modules must
import and expose their documented symbols.
"""
from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import coordpy  # noqa: E402
from coordpy import subject as S  # noqa: E402

REGISTRY_DOC = os.path.join(ROOT, "docs", "W144_COORDPY_SUBJECT_REGISTRY.json")


# --- stable boundary invariant -------------------------------------------

def test_stable_boundary_invariant():
    assert coordpy.__version__ == "0.5.20"
    assert coordpy.SDK_VERSION == "coordpy.sdk.v3.43"
    assert S.SUBJECT_REPORT_SCHEMA == "coordpy.subject.v1"
    assert S.EXPECTED_VERSION == coordpy.__version__
    assert S.EXPECTED_SDK_VERSION == coordpy.SDK_VERSION


# --- harness checks all PASS ---------------------------------------------

def test_run_harness_all_pass():
    checks = S.run_harness()
    assert len(checks) == 4
    names = {c["name"] for c in checks}
    assert names == {"stable_smoke", "team_runtime", "capsule_verify", "s2_exemplars"}
    for c in checks:
        assert c["status"] == "PASS", f"{c['name']} failed: {c['detail']}"


def test_capsule_verify_is_load_bearing():
    """The capsule_verify check must depend on a real sealed team view,
    i.e. a None view fails rather than silently passing."""
    assert S.check_capsule_verify(None)["status"] == "FAIL"
    _team, view = S.check_team_runtime()
    assert view is not None and view.get("schema") == "coordpy.capsule_view.v1"
    assert S.check_capsule_verify(view)["status"] == "PASS"


def test_stable_smoke_detects_missing_symbol(monkeypatch):
    """The smoke check is a real gate: hide a stable symbol -> FAIL."""
    monkeypatch.delattr(coordpy, "AgentTeam", raising=True)
    assert S.check_stable_smoke()["status"] == "FAIL"


# --- report + tier map ----------------------------------------------------

def test_build_subject_report_shape():
    rep = S.build_subject_report(run_checks=True)
    assert rep["schema"] == "coordpy.subject.v1"
    for key in ("subject", "front_doors", "tiers", "out_of_scope", "registry_doc"):
        assert key in rep
    assert rep["all_pass"] is True
    # mechanism-shaped centre of gravity, not outcome-shaped
    assert "capsule" in rep["subject"]["one_line"].lower()
    assert set(rep["subject"]["capsule_contract"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}


def test_report_no_checks_is_pure_description():
    rep = S.build_subject_report(run_checks=False)
    assert "checks" not in rep and "all_pass" not in rep


def test_all_five_tiers_present():
    tiers = S.TIERS
    assert list(tiers) == [
        "S1_stable_core", "S2_canonical_experimental",
        "S3_benchmark_research_support", "S4_historical_archive",
        "S5_blocked_dead_keep_for_record",
    ]
    for t in tiers.values():
        assert "definition" in t and t["definition"]


def test_s1_matches_documented_stable_surface():
    s1 = S.TIERS["S1_stable_core"]
    # the load-bearing stable modules must be present
    for m in ("capsule", "capsule_runtime", "run", "runtime", "agents",
              "team_coord", "presets", "llm_backend", "lifecycle_audit",
              "extensions", "subject"):
        assert m in s1["modules"], f"{m} missing from S1"
    # both front doors enumerated among the console scripts
    scripts = " ".join(s1["console_scripts"])
    assert "coordpy-team" in scripts and "coordpy-subject" in scripts
    # the canonical schemas
    assert "coordpy.capsule_view.v1" in s1["schemas"]
    assert "coordpy.subject.v1" in s1["schemas"]


def test_front_door_decision():
    fd = S.FRONT_DOORS
    assert fd["usage"]["entrypoint"] == "coordpy-team"
    assert fd["orientation"]["entrypoint"] == "coordpy-subject"


# --- S2 canonical-experimental modules import (closes the coverage gap) ---

@pytest.mark.parametrize("mod_name,symbol", S._S2_EXEMPLARS)
def test_s2_exemplar_imports(mod_name, symbol):
    mod = importlib.import_module(mod_name)
    assert hasattr(mod, symbol), f"{mod_name} missing documented symbol {symbol}"


def test_untested_chain_modules_now_covered():
    """The three W140..W143 chain modules that previously had no dedicated
    test now import + expose their documented entry symbols."""
    from coordpy.self_tutoring_controller_v1 import discover_self_scaffold_v1
    from coordpy.no_oracle_verifier_v2 import select_winner_v2
    from coordpy.multi_agent_discover_amortize_v1 import team_discover_v1
    assert callable(discover_self_scaffold_v1)
    assert callable(select_winner_v2)
    assert callable(team_discover_v1)


# --- registry doc <-> embedded registry ----------------------------------

def test_registry_doc_matches_embedded():
    assert os.path.exists(REGISTRY_DOC), "registry doc missing"
    with open(REGISTRY_DOC, encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["schema"] == "coordpy.subject.v1"
    # the doc mirrors the embedded tier map (the harness is the source of truth)
    assert list(doc["tiers"]) == list(S.TIERS)
    assert doc["front_doors"]["usage"]["entrypoint"] == "coordpy-team"
    assert doc["front_doors"]["orientation"]["entrypoint"] == "coordpy-subject"
    assert doc["subject"]["one_line"] == S.CANONICAL_SUBJECT["one_line"]


# --- CLI exit-code contract ----------------------------------------------

def test_cli_report_and_check_exit_zero_on_pass():
    assert S.main(["report"]) == 0
    assert S.main(["report", "--json"]) == 0
    assert S.main(["check"]) == 0
    assert S.main(["check", "--json"]) == 0
    assert S.main(["registry"]) == 0
    assert S.main(["tiers"]) == 0
    assert S.main(["--version"]) == 0
    assert S.main([]) == 0  # default == report


def test_cli_main_subject_wrapper_importable():
    from coordpy._cli import main_subject
    assert callable(main_subject)
