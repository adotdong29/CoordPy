"""W124 Lane α — contract tests for the transformer-native code-intervention
mechanism module. Falsifiability-first: the probe must SEPARATE genuinely
separable data and must NOT manufacture signal on noise; the gate must refuse a
close blip; the M6 controller must route deterministically without ever reading
a hidden test.

Runs under any env that has numpy+sklearn (the real distilgpt2 encoder is NOT
exercised here — that is the bench's job). Heavy deps are import-skipped.
"""
from __future__ import annotations

# Resilient imports: prefer pytest's importorskip, but stay runnable as a plain
# script when the local pytest is broken (the opt/anaconda3 env has an
# attrs/pytest incompat) so the suite can be validated via direct execution.
try:
    import pytest
    mod = pytest.importorskip("coordpy.transformer_native_code_intervention_v1")
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
except Exception:  # noqa: BLE001
    import importlib
    mod = importlib.import_module("coordpy.transformer_native_code_intervention_v1")
    import numpy as np
    import sklearn  # noqa: F401


# ----- surface features -----
def test_surface_features_fixed_length_and_deterministic():
    a = mod.surface_features_v1("def f():\n    return 1\n")
    b = mod.surface_features_v1("def f():\n    return 1\n")
    c = mod.surface_features_v1("x = 1\n")
    assert a == b
    assert len(a) == len(c) == 11
    assert all(isinstance(v, float) for v in a)


def test_surface_features_handles_unparseable_code():
    v = mod.surface_features_v1("def (((")  # syntax error
    assert v[8] == 0.0  # parses flag = 0
    assert len(v) == 11


# ----- stub encoder (no torch) -----
def test_stub_encoder_runs_without_torch():
    enc = mod.AstBoundaryHiddenEncoderV1(model_name="stub", require_real_model=False)
    v1 = enc.encode("def solve():\n    return 1\n")
    v2 = enc.encode("def solve():\n    return 1\n")
    assert v1.shape == (16,)
    assert np.allclose(v1, v2)  # deterministic + cached
    assert "stub" in enc.encoder_id()


def _rows_from(sources_labels, field, probmaker):
    out = []
    for i, (src, lab) in enumerate(sources_labels):
        out.append(mod.LabeledCodeRowV1(
            problem_id=probmaker(i), field=field, arm="A1", source=src,
            passed=bool(lab), label_source="synthetic"))
    return out


# ----- probe: separable vs noise (stub encoder, sklearn) -----
def test_probe_detects_separable_then_gate_real_signal():
    # Build a dataset where the ENCODER feature is perfectly predictive but the
    # SURFACE feature is not: positives are long, negatives short ... but we make
    # surface non-informative by equalizing length, and inject a separable hidden
    # vector via a fake encoder.
    rng = np.random.RandomState(0)

    class FakeEnc:
        def encoder_id(self):
            return "fake:separable"

        def encode(self, source):
            # encode the label hidden in a marker comment, plus noise
            base = 3.0 if "PASS" in source else -3.0
            return (np.array([base], dtype=np.float32) + rng.randn(1).astype(np.float32) * 0.1)

    rows = []
    for p in range(12):  # 12 problems, each 6 samples, mixed labels within problem
        for s in range(6):
            lab = 1 if (s % 2 == 0) else 0
            # equal-length sources so surface baseline cannot separate
            src = ("# PASS\n" if lab else "# fail\n") + "def f():\n    return 0\n"
            rows.append(mod.LabeledCodeRowV1(
                problem_id=f"resistant:{p}", field=("resistant" if p % 2 else "exposed"),
                arm="A1", source=src, passed=bool(lab), label_source="synthetic"))
    rep = mod.run_m4_separability_probe_v1(rows, encoder=FakeEnc(), n_splits=4, seed=1)
    assert rep.auc_hidden > 0.85           # hidden cleanly separates
    assert rep.auc_hidden - rep.auc_surface >= 0.05
    gate = mod.m4_gate_v1(rep)
    assert gate["verdict"] == "M4_REAL_SIGNAL"
    assert gate["real_signal"] is True


def test_probe_finds_no_signal_on_noise_then_gate_signal_poor():
    rng = np.random.RandomState(7)

    class NoiseEnc:
        def encoder_id(self):
            return "fake:noise"

        def encode(self, source):
            return rng.randn(8).astype(np.float32)  # label-independent noise

    rows = []
    for p in range(12):
        for s in range(6):
            lab = 1 if (s % 3 == 0) else 0
            rows.append(mod.LabeledCodeRowV1(
                problem_id=f"p{p}", field=("resistant" if p % 2 else "exposed"),
                arm="A1", source=f"def f{p}_{s}():\n    return {s}\n",
                passed=bool(lab), label_source="synthetic"))
    rep = mod.run_m4_separability_probe_v1(rows, encoder=NoiseEnc(), n_splits=4, seed=2)
    gate = mod.m4_gate_v1(rep)
    assert gate["verdict"] in ("M4_SIGNAL_POOR", "M4_CLOSE_BLIP_NOT_A_GAIN")
    assert gate["real_signal"] is False


# ----- gate arithmetic (pre-committed thresholds) -----
def _mk_report(ah, as_, ahr=None, ahe=None, asr=None, ase=None):
    return mod.M4SeparabilityReportV1(
        schema="t", n_rows=100, n_pos=30, n_groups=10, encoder_id="t",
        auc_hidden=ah, auc_surface=as_,
        auc_hidden_resistant=ah if ahr is None else ahr,
        auc_hidden_exposed=ah if ahe is None else ahe,
        auc_surface_resistant=as_ if asr is None else asr,
        auc_surface_exposed=as_ if ase is None else ase,
        n_splits=5, label_source_counts={})


def test_gate_thresholds():
    assert mod.m4_gate_v1(_mk_report(0.70, 0.55))["verdict"] == "M4_REAL_SIGNAL"
    # below AUC floor even if margin big
    assert mod.m4_gate_v1(_mk_report(0.58, 0.40))["verdict"] == "M4_CLOSE_BLIP_NOT_A_GAIN"
    # above floor but margin < 0.05
    assert mod.m4_gate_v1(_mk_report(0.62, 0.60))["verdict"] == "M4_CLOSE_BLIP_NOT_A_GAIN"
    # hidden below surface
    assert mod.m4_gate_v1(_mk_report(0.50, 0.55))["verdict"] == "M4_SIGNAL_POOR"
    # margin ok + floor ok but sign-inconsistent across slices -> not real
    g = mod.m4_gate_v1(_mk_report(0.70, 0.55, ahr=0.50, asr=0.65))
    assert g["verdict"] != "M4_REAL_SIGNAL"


def test_report_cid_deterministic():
    r1 = _mk_report(0.61, 0.60)
    r2 = _mk_report(0.61, 0.60)
    assert r1.cid() == r2.cid()
    assert r1.cid() != _mk_report(0.62, 0.60).cid()


# ----- M5 gated off when M4 not real -----
def test_m5_not_run_when_m4_signal_poor():
    gate = {"real_signal": False}
    out = mod.run_m5_projector_if_earned_v1([], encoder=None, m4_gate=gate)
    assert out["status"] == "NOT_RUN_M4_SIGNAL_POOR"
    assert out["hosted_translatable"] is False


# ----- M6 deterministic controller, never reads hidden tests -----
def test_m6_controller_routes_deterministically():
    c = mod.M6ToolSubstrateCodeControllerV1(k_budget=5)
    assert c.route(stderr_tail="SyntaxError: bad", timed_out=False, attempt_idx=0).route == mod.M6_ROUTE_PATCH
    assert c.route(stderr_tail="", timed_out=True, attempt_idx=1).route == mod.M6_ROUTE_REPLAN
    # budget exhausted -> abstain
    assert c.route(stderr_tail="ValueError: x", timed_out=False, attempt_idx=4).route == mod.M6_ROUTE_ABSTAIN
    # determinism
    d1 = c.route(stderr_tail="IndexError: y", timed_out=False, attempt_idx=1)
    d2 = c.route(stderr_tail="IndexError: y", timed_out=False, attempt_idx=1)
    assert d1 == d2
    assert c.is_hosted_translatable() is True


def test_m6_contract_reads_no_hidden_tests():
    # the controller only consumes a stderr tail + timeout flag + attempt idx
    import inspect
    sig = inspect.signature(mod.M6ToolSubstrateCodeControllerV1.route)
    assert set(sig.parameters) - {"self"} == {"stderr_tail", "timed_out", "attempt_idx"}


# ----- verdict assembly: hosted NOT earned when M4 signal-poor -----
def test_verdict_no_hosted_when_signal_poor():
    rep = _mk_report(0.50, 0.50)
    gate = mod.m4_gate_v1(rep)
    v = mod.assemble_lane_alpha_verdict_v1(
        m4=rep, m4_gate=gate,
        m5={"status": "NOT_RUN_M4_SIGNAL_POOR", "hosted_translatable": False},
        m6_contract={"hosted_translatable": True, "local_gain_demonstrated": False},
        dataset_manifest={})
    assert v["hosted_maverick_probe_earned"] is False
    assert v["candidates_survived"] == []


if __name__ == "__main__":
    # Direct-execution fallback (runnable without pytest). Runs every test_*.
    import traceback
    fns = sorted((n, f) for n, f in globals().items()
                 if n.startswith("test_") and callable(f))
    passed = failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS", name)
            passed += 1
        except Exception:  # noqa: BLE001
            print("FAIL", name)
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    raise SystemExit(1 if failed else 0)
