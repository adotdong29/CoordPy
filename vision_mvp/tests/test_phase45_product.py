"""Phase-45 — product runner tests.

Covers profile validity, readiness-only flows, mock-oracle sweep
integration, skip-sweep, and the public_jsonl missing-path guard.
"""

from __future__ import annotations

import json
import os
import tempfile

from vision_mvp.product import profiles
from vision_mvp.product.runner import run_profile
from vision_mvp.product.report import render_summary


def test_profiles_known_set():
    names = profiles.list_profiles()
    for required in ("local_smoke", "bundled_57",
                      "bundled_57_mock_sweep",
                      "aspen_mac1_coder", "aspen_mac2_frontier",
                      "public_jsonl"):
        assert required in names, names


def test_profile_deepcopy_is_independent():
    a = profiles.get_profile("local_smoke")
    b = profiles.get_profile("local_smoke")
    a["readiness"]["limit"] = 999
    assert b["readiness"]["limit"] == 8


def test_local_smoke_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile("local_smoke", out_dir=td)
        assert rep["readiness"]["ready"] is True
        assert rep["readiness"]["n"] == 8
        assert rep["readiness"]["n_passed_all"] == 8
        assert rep["sweep"]["mode"] == "mock"
        # 2 parser modes × 1 apply × 1 nd = 2 cells.
        assert len(rep["sweep"]["cells"]) == 2
        for cell in rep["sweep"]["cells"]:
            pooled = cell["pooled"]
            for strat in ("naive", "routing", "substrate"):
                assert pooled[strat]["pass_at_1"] == 1.0
        # Artifacts on disk.
        assert os.path.exists(os.path.join(td, "product_report.json"))
        assert os.path.exists(os.path.join(td, "product_summary.txt"))
        assert os.path.exists(os.path.join(td, "readiness_verdict.json"))
        assert os.path.exists(os.path.join(td, "sweep_result.json"))


def test_bundled_57_readiness_only_full_saturation():
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile("bundled_57", out_dir=td)
        assert rep["readiness"]["ready"] is True
        assert rep["readiness"]["n"] == 57
        assert rep["readiness"]["n_passed_all"] == 57
        assert rep["sweep"] is None
        for (name, r) in rep["readiness"]["checks"].items():
            assert r["failed"] == 0, (name, r)


def test_aspen_real_profile_records_launch_cmd_without_executing():
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile("aspen_mac1_coder", out_dir=td)
        assert rep["readiness"]["ready"] is True
        sw = rep["sweep"]
        assert sw["mode"] == "real"
        assert sw["executed_in_process"] is False
        # Launch cmd must reference the correct experiment module
        # and the macbook-1 URL.
        assert "phase42_parser_sweep" in " ".join(sw["launch_cmd"])
        assert "192.168.12.191" in " ".join(sw["launch_cmd"])
        cap = sw.get("raw_capture_launch_cmd") or []
        assert "phase44_semantic_residue" in " ".join(cap)


def test_skip_sweep_flag_skips_sweep_block():
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile(
            "local_smoke", out_dir=td, skip_sweep=True)
        assert rep["readiness"]["ready"] is True
        assert rep["sweep"] is None


def test_public_jsonl_profile_requires_override():
    try:
        with tempfile.TemporaryDirectory() as td:
            run_profile("public_jsonl", out_dir=td)
    except SystemExit as ex:
        assert "jsonl" in str(ex)
        return
    raise AssertionError("expected SystemExit for public_jsonl w/o --jsonl")


def test_public_jsonl_profile_accepts_override():
    # Untrusted profile — require explicit --allow-unsafe-sandbox
    # on machines without Docker (test env is not required to have
    # a Docker daemon).
    bundled = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile(
            "public_jsonl", out_dir=td, jsonl_override=bundled,
            allow_unsafe_sandbox=True)
        assert rep["readiness"]["ready"] is True


def test_public_jsonl_profile_is_untrusted_and_docker_first():
    from vision_mvp.product import profiles as _pf
    prof = _pf.get_profile("public_jsonl")
    assert prof["trust"] == _pf.TRUST_UNTRUSTED
    assert prof["readiness"]["sandbox_name"] == "docker"
    assert _pf.is_untrusted("public_jsonl") is True
    assert _pf.is_untrusted("local_smoke") is False


def test_public_jsonl_refuses_weak_sandbox_without_docker():
    # Simulate "Docker unavailable" by patching DockerSandbox.is_available.
    from vision_mvp.tasks import swe_sandbox
    bundled = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    orig = swe_sandbox.DockerSandbox.is_available
    swe_sandbox.DockerSandbox.is_available = lambda self: False
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                run_profile("public_jsonl", out_dir=td,
                             jsonl_override=bundled)
            except SystemExit as ex:
                msg = str(ex)
                assert "Docker" in msg or "docker" in msg
                assert "allow-unsafe-sandbox" in msg
                return
        raise AssertionError(
            "expected SystemExit when Docker unavailable and "
            "allow_unsafe_sandbox=False")
    finally:
        swe_sandbox.DockerSandbox.is_available = orig


def test_public_jsonl_allow_unsafe_downgrades_to_subprocess():
    from vision_mvp.tasks import swe_sandbox
    bundled = os.path.join(
        os.path.dirname(__file__), "..", "tasks", "data",
        "swe_lite_style_bank.jsonl")
    orig = swe_sandbox.DockerSandbox.is_available
    swe_sandbox.DockerSandbox.is_available = lambda self: False
    try:
        with tempfile.TemporaryDirectory() as td:
            rep = run_profile(
                "public_jsonl", out_dir=td, jsonl_override=bundled,
                allow_unsafe_sandbox=True)
            assert rep["readiness"]["ready"] is True
    finally:
        swe_sandbox.DockerSandbox.is_available = orig


def test_render_summary_has_required_fields():
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile("local_smoke", out_dir=td)
        text = render_summary(rep)
        for phrase in ("profile", "readiness", "sweep",
                        "local_smoke", "READY"):
            assert phrase in text


def test_model_capability_table_lists_canonical_models():
    tab = profiles.model_capability_table()
    assert "qwen2.5-coder:14b" in tab
    assert "qwen3.5:35b" in tab
    assert "semantic_headroom" in tab["qwen3.5:35b"]
    assert "canonical_coder" in tab["qwen2.5-coder:14b"]


def test_report_schema_stable():
    # v2 is the current stable product-report schema (Slice 2+).
    # v1 is still accepted by the CI gate for backwards-compat —
    # see ``vision_mvp/product/ci_gate.py::EXPECTED_REPORT_SCHEMAS``.
    with tempfile.TemporaryDirectory() as td:
        rep = run_profile("local_smoke", out_dir=td)
        with open(os.path.join(td, "product_report.json"),
                    "r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["schema"] == "phase45.product_report.v2"
        assert on_disk["profile"] == "local_smoke"
        assert "readiness" in on_disk
        assert "sweep" in on_disk
