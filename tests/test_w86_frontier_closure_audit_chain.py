"""W86 — verify the frontier-closure audit chain on disk.

Runs ``scripts/verify_w86_audit_chain.py`` against the captured
results in ``results/w86/w86_20260520T012814Z/`` — the third
independent Colab Pro A100 run on 2026-05-20. The audit chain is
the load-bearing third-party-re-verifiability claim for the W86
#25 + #26 closure; if the recorded CIDs don't re-derive from the
canonical JSON bytes, the closure is not honest.

This is a CI-time check: no torch / transformers needed.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


W86_RUN3_DIR = (
    Path(__file__).resolve().parent.parent
    / "results" / "w86" / "w86_20260520T012814Z")


def _canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode("utf-8")


def _sha256_hex(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "frontier_closure_report.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_top_level_report_cid_re_derives():
    """The top-level frontier_closure_report.json must re-hash
    to its recorded ``report_cid``."""
    report = json.loads(
        (W86_RUN3_DIR / "frontier_closure_report.json")
        .read_bytes().decode("utf-8"))
    recorded = report.get("report_cid", "")
    derived = _sha256_hex({
        "kind": "w86_frontier_closure_report_v1",
        "report": {
            k: v for k, v in report.items()
            if k != "report_cid"},
    })
    assert derived == recorded, (
        f"recorded={recorded} derived={derived}")


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "25_substrate_coupling.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_closure_25_sidecar_cid_re_derives():
    side = json.loads(
        (W86_RUN3_DIR / "25_substrate_coupling.json")
        .read_bytes().decode("utf-8"))
    recorded = side.get("report_cid", "")
    derived = _sha256_hex({
        "kind": "w86_25_substrate_coupling_report",
        "out": {
            k: v for k, v in side.items()
            if k != "report_cid"},
    })
    assert derived == recorded


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "26_live_learned_memory.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_closure_26_sidecar_cid_re_derives():
    side = json.loads(
        (W86_RUN3_DIR / "26_live_learned_memory.json")
        .read_bytes().decode("utf-8"))
    recorded = side.get("report_cid", "")
    derived = _sha256_hex({
        "kind": "w86_26_live_learned_memory_report",
        "out": {
            k: v for k, v in side.items()
            if k != "report_cid"},
    })
    assert derived == recorded


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "frontier_closure_report.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_25_load_bearing_claims():
    """The #25 substrate coupling closure must report:
    n_pass >= 10 of 12 axes, hidden_state_intercept_moves_cid =
    True, replay_byte_identical at the bf16 tier = True, and the
    W83 load-bearing claim reproduced at frontier scale = True.
    """
    report = json.loads(
        (W86_RUN3_DIR / "frontier_closure_report.json")
        .read_bytes().decode("utf-8"))
    c25 = report.get("closure_25", {})
    assert c25.get("issue") == 25
    conf = c25.get("conformance", {})
    assert int(conf.get("n_pass", 0)) >= 10
    hib = c25.get("hidden_state_intercept_bench", {})
    assert hib.get("hidden_state_intercept_moves_cid") is True
    assert hib.get("replay_byte_identical") is True
    rkv = c25.get("replay_from_kv", {})
    assert rkv.get("replay_byte_identical") is True
    assert rkv.get("precision_tier") == "tier_bf16"
    assert float(rkv.get(
        "max_abs_diff_last_logits", 1.0)) < float(rkv.get(
            "precision_tier_tolerance", 0.0))
    assert c25.get("w83_load_bearing_claim_reproduced") is True


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "frontier_closure_report.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_26_strict_beat():
    """The #26 live-learned-memory closure must report
    live_strictly_beats_synthetic_on_holdout = True with live
    MSE strictly < synthetic MSE."""
    report = json.loads(
        (W86_RUN3_DIR / "frontier_closure_report.json")
        .read_bytes().decode("utf-8"))
    c26 = report.get("closure_26", {})
    assert c26.get("issue") == 26
    assert c26.get("live_strictly_beats_synthetic") is True
    tr = c26.get("train_report", {})
    live = float(tr.get("live_mse_on_holdout_live", 1e9))
    syn = float(tr.get("synthetic_mse_on_holdout_live", 0.0))
    assert live < syn, (
        f"live MSE {live} must be strictly < syn MSE {syn}")


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "frontier_closure_report.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_verifier_script_passes_offline():
    """``scripts/verify_w86_audit_chain.py`` must exit 0 on the
    third-run evidence — the audit chain is offline-re-verifiable
    by a third party."""
    repo_root = Path(__file__).resolve().parent.parent
    report_path = (
        W86_RUN3_DIR / "frontier_closure_report.json")
    proc = subprocess.run(
        [sys.executable,
         str(repo_root / "scripts" / "verify_w86_audit_chain.py"),
         "--report", str(report_path)],
        capture_output=True, text=True, check=False)
    assert proc.returncode == 0, (
        f"verifier failed: {proc.stdout}\n{proc.stderr}")
    # All four DoD bars on #25 and the #26 strict-beat must
    # appear as PASS.
    assert "PASS #25 hidden-state intercept moves CID" in (
        proc.stdout)
    assert "PASS #25 W80 conformance" in proc.stdout
    assert "PASS #25 replay-from-KV at tier" in proc.stdout
    assert "PASS #26 live-trained MSE strictly" in proc.stdout


@pytest.mark.skipif(
    not (W86_RUN3_DIR / "frontier_closure_report.json").exists(),
    reason="W86 run-3 evidence not present in this checkout")
def test_w86_run3_model_identity():
    """The report must record the exact model name used —
    anti-cheat: no silent swap to a smaller/different model."""
    report = json.loads(
        (W86_RUN3_DIR / "frontier_closure_report.json")
        .read_bytes().decode("utf-8"))
    assert report.get(
        "model_name") == "meta-llama/Llama-3.1-8B-Instruct"
    assert int(report.get(
        "model_n_layers", 0)) == 32
    assert int(report.get(
        "model_n_heads", 0)) == 32
    assert int(report.get(
        "model_hidden_dim", 0)) == 4096
    assert report.get("precision_tier") == "tier_bf16"
    assert report.get("device") == "cuda:0"
