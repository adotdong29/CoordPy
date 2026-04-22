"""Phase-46 — public-data import, CI gate, frontier-slot profile.

Covers:
  * ``import_data.audit_jsonl`` on the bundled bank (clean path).
  * ``import_data.audit_jsonl`` on a constructed broken JSONL
    (missing keys, decode error, non-object row, duplicate ids).
  * ``import_data.audit_jsonl`` on a missing path.
  * ``ci_gate.evaluate_report`` on each of the three Phase-45 RC
    artifacts (bundled_57, bundled_57_mock_sweep, aspen mac1).
  * ``ci_gate`` profile-whitelist rejection.
  * Frontier-slot profile (``aspen_mac1_coder_70b``) + the
    ``model_availability`` declarative check.
"""

from __future__ import annotations

import json
import os
import tempfile

from vision_mvp.product import profiles
from vision_mvp.product.import_data import audit_jsonl
from vision_mvp.product.ci_gate import evaluate_report
from vision_mvp.product.runner import run_profile


_BUNDLED = os.path.join(
    os.path.dirname(__file__), "..", "tasks", "data",
    "swe_lite_style_bank.jsonl")


# ---------- import_data ----------

def test_import_bundled_bank_is_ready():
    rep = audit_jsonl(_BUNDLED, limit=8, sandbox_name="in_process")
    assert rep["ok"] is True
    assert rep["n_rows"] == 8
    assert rep["blockers"] == []
    # Bundled bank rows carry both native + hermetic keys.
    assert rep["shape_counts"].get("ambiguous", 0) >= 1
    assert rep["readiness"] is not None
    assert rep["readiness"]["ready"] is True


def test_import_missing_file_returns_file_not_found():
    rep = audit_jsonl("/no/such/path.jsonl")
    assert rep["ok"] is False
    assert rep["error_kind"] == "file_not_found"


def test_import_broken_jsonl_surfaces_blockers():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "broken.jsonl")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"instance_id": "x"}) + "\n")
            fh.write("{not valid json\n")
            fh.write(json.dumps(["array", "not", "object"]) + "\n")
            fh.write(json.dumps({"instance_id": "x",
                                   "patch": "p", "repo": "r",
                                   "base_commit": "b",
                                   "problem_statement": "ps"}) + "\n")
        rep = audit_jsonl(p, run_readiness_check=False)
        assert rep["ok"] is False
        assert rep["n_rows"] == 4
        kinds = " ".join(rep["blockers"])
        assert "json_decode_errors" in kinds
        assert "non_object_rows" in kinds
        assert "unusable_rows" in kinds
        assert "duplicate_instance_ids" in kinds


def test_import_empty_bank_is_blocker():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "empty.jsonl")
        open(p, "w").close()
        rep = audit_jsonl(p, run_readiness_check=False)
        assert rep["ok"] is False
        assert "empty_bank" in rep["blockers"]


# ---------- ci_gate ----------

def _make_rc_bundle(td: str) -> str:
    rep = run_profile("bundled_57", out_dir=td)
    assert rep["readiness"]["ready"]
    return os.path.join(td, "product_report.json")


def test_ci_gate_accepts_bundled_57_rc():
    with tempfile.TemporaryDirectory() as td:
        path = _make_rc_bundle(td)
        v = evaluate_report(path)
        assert v["ok"] is True, v["blockers"]
        assert v["checks"]["schema"]["ok"]
        assert v["checks"]["profile"]["ok"]
        assert v["checks"]["readiness"]["ok"]
        assert v["checks"]["artifacts"]["ok"]


def test_ci_gate_rejects_non_whitelisted_profile():
    with tempfile.TemporaryDirectory() as td:
        path = _make_rc_bundle(td)
        v = evaluate_report(
            path, require_profiles=("aspen_mac2_frontier",))
        assert v["ok"] is False
        assert any(b.startswith("profile_not_allowed")
                    for b in v["blockers"])


def test_ci_gate_requires_sweep_executed_fails_on_readiness_only():
    with tempfile.TemporaryDirectory() as td:
        path = _make_rc_bundle(td)
        v = evaluate_report(path, require_sweep_executed=True)
        assert v["ok"] is False
        assert any("sweep" in b for b in v["blockers"])


def test_ci_gate_local_smoke_sweep_passes_threshold():
    with tempfile.TemporaryDirectory() as td:
        run_profile("local_smoke", out_dir=td)
        v = evaluate_report(
            os.path.join(td, "product_report.json"),
            min_pass_at_1=1.0)
        assert v["ok"] is True, v["blockers"]
        assert v["checks"]["sweep"]["ok"]


def test_ci_gate_missing_report_blocks():
    v = evaluate_report("/nope/product_report.json")
    assert v["ok"] is False
    assert "report_not_found" in v["blockers"]


# ---------- frontier slot ----------

def test_frontier_slot_profile_is_registered():
    assert "aspen_mac1_coder_70b" in profiles.list_profiles()
    prof = profiles.get_profile("aspen_mac1_coder_70b")
    assert prof["requires_model_availability"] is True
    assert prof["sweep"]["model"].endswith(":70b")
    assert prof["sweep"]["ollama_url"].endswith(":11434")


def test_model_availability_declarative():
    assert (profiles.model_availability("qwen2.5-coder:70b")
            ["availability"] == "pending_availability")
    assert (profiles.model_availability("qwen2.5-coder:14b")
            ["availability"] == "assumed_resident")
    assert profiles.model_availability(None)["availability"] == "n/a"


def test_frontier_slot_sweep_metadata_attached_in_report():
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile("aspen_mac1_coder_70b", out_dir=td)
        sw = rep["sweep"]
        assert sw["mode"] == "real"
        assert sw["executed_in_process"] is False
        assert sw["model_metadata"]["availability"] == (
            "pending_availability")
