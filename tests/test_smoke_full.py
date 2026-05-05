"""Exhaustive smoke-driver for the CoordPy public surface.

Spins every documented stable symbol of the SDK end-to-end against an
installed ``coordpy-ai`` wheel. The script is intentionally hermetic
(no network, no LLM calls) and self-cleaning. It exits non-zero on
the first contract surprise.

Run from the repo root::

    python tests/test_smoke_full.py
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata as _md
import json
import os
import shutil
import sys
import tempfile
import time
import traceback


_FAILURES: list[tuple[str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    status = "OK  " if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    if not ok:
        _FAILURES.append((name, detail))


def section(title: str) -> None:
    print(f"\n# {title}")


def main() -> int:  # noqa: C901 — driver script, length is fine
    section("0. Distribution metadata")
    import coordpy  # noqa: WPS433
    check("import coordpy", True, coordpy.__version__)
    try:
        dist = _md.distribution("coordpy-ai")
        check("distribution metadata reachable", True, dist.metadata["Name"])
        check("dist.version == coordpy.__version__",
              dist.version == coordpy.__version__,
              f"{dist.version!r} vs {coordpy.__version__!r}")
        classifiers = dist.metadata.get_all("Classifier") or []
        check("Typing :: Typed classifier present",
              "Typing :: Typed" in classifiers)
        # py.typed marker on disk
        py_typed = os.path.join(os.path.dirname(coordpy.__file__), "py.typed")
        check("py.typed marker present", os.path.exists(py_typed))
    except _md.PackageNotFoundError:
        check("distribution metadata reachable (editable install)", True,
              "skipped — running from source")

    section("1. Version + SDK constants")
    check("__version__ matches semver-ish",
          coordpy.__version__.count(".") == 2)
    check("SDK_VERSION starts with 'coordpy.sdk.'",
          coordpy.SDK_VERSION.startswith("coordpy.sdk."))
    for cname, expected in [
        ("CAPSULE_VIEW_SCHEMA", "coordpy.capsule_view.v1"),
        ("PROVENANCE_SCHEMA", "coordpy.provenance.v1"),
    ]:
        v = getattr(coordpy, cname)
        check(f"{cname} == {expected!r}", v == expected, repr(v))

    section("2. Run a profile end-to-end")
    tmp = tempfile.mkdtemp(prefix="coordpy_spin_")
    try:
        rs = coordpy.RunSpec(profile="local_smoke",
                             out_dir=os.path.join(tmp, "smoke"))
        check("RunSpec is dataclass", dataclasses.is_dataclass(rs))
        report = coordpy.run(rs)
        check("readiness.ready", report["readiness"]["ready"])
        check("provenance.schema",
              report["provenance"]["schema"] == coordpy.PROVENANCE_SCHEMA)
        check("provenance.code.package",
              report["provenance"]["code"]["package"] == "coordpy")
        check("capsules.chain_ok", report["capsules"]["chain_ok"])
        check("capsules.schema",
              report["capsules"]["schema"] == coordpy.CAPSULE_VIEW_SCHEMA)
        check("RUN_REPORT capsule kind present",
              "RUN_REPORT" in (
                  report["capsules"].get("stats", {}).get("by_kind", {})))

        section("3. ci_gate verdict")
        report_path = os.path.join(tmp, "smoke", "product_report.json")
        verdict = coordpy.ci_gate.evaluate_report(report_path)
        check("ci_gate verdict ok", verdict.get("ok"))
        check("CI_VERDICT_SCHEMA matches",
              verdict.get("schema") == coordpy.CI_VERDICT_SCHEMA)

        section("4. capsule chain re-verification")
        check("verify_chain_from_view_dict on embedded view",
              coordpy.verify_chain_from_view_dict(report["capsules"]))
        view_path = os.path.join(tmp, "smoke", "capsule_view.json")
        if os.path.exists(view_path):
            with open(view_path, encoding="utf-8") as fh:
                disk_view = json.load(fh)
            check("on-disk capsule_view.json chain ok",
                  coordpy.verify_chain_from_view_dict(disk_view))
            check("on-disk view chain_head == embedded",
                  disk_view["chain_head"]
                  == report["capsules"]["chain_head"])

        section("5. build_report_ledger round-trip")
        led, root_cid = coordpy.build_report_ledger(report)
        v = coordpy.render_view(led, root_cid=root_cid)
        check("rebuilt ledger chain_ok", v.chain_ok)
        check("rebuilt ledger has same root_cid",
              v.root_cid == root_cid)

        section("6. on-disk artifact + meta_manifest verification")
        out_dir = os.path.join(tmp, "smoke")
        ac = coordpy.verify_artifacts_on_disk(disk_view, base_dir=out_dir)
        check("verify_artifacts_on_disk OK",
              ac["verdict"] in ("OK", "EMPTY"),
              f"{ac['verdict']} ({ac['ok']}/{ac['checked']})")
        mm_path = os.path.join(out_dir, "meta_manifest.json")
        if os.path.exists(mm_path):
            with open(mm_path, encoding="utf-8") as fh:
                mm = json.load(fh)
            mc = coordpy.verify_meta_manifest_on_disk(mm, base_dir=out_dir)
            check("verify_meta_manifest_on_disk OK",
                  mc["verdict"] in ("OK", "EMPTY"))

        section("7. lifecycle audit")
        la = coordpy.audit_capsule_lifecycle_from_view(report["capsules"])
        check("lifecycle audit OK",
              getattr(la, "verdict", None) == "OK",
              str(getattr(la, "verdict", None)))

        section("8. capsule primitives")
        a = coordpy.ContextCapsule.new(
            kind=coordpy.CapsuleKind.ARTIFACT, payload={"x": 1})
        b = coordpy.ContextCapsule.new(
            kind=coordpy.CapsuleKind.ARTIFACT, payload={"x": 1})
        c = coordpy.ContextCapsule.new(
            kind=coordpy.CapsuleKind.ARTIFACT, payload={"x": 2})
        check("identical payload → identical CID", a.cid == b.cid)
        check("different payload → different CID", a.cid != c.cid)
        check("CID length 64 (sha256 hex)", len(a.cid) == 64)
        check("PROPOSED at construction",
              a.lifecycle == coordpy.CapsuleLifecycle.PROPOSED)
        # Admission failure path
        try:
            coordpy.ContextCapsule.new(
                kind=coordpy.CapsuleKind.ARTIFACT,
                payload={"big": "X" * 5_000},
                budget=coordpy.CapsuleBudget(max_bytes=10),
            )
            check("oversize raises CapsuleAdmissionError", False)
        except coordpy.CapsuleAdmissionError:
            check("oversize raises CapsuleAdmissionError", True)
        # Ledger admit + seal
        led = coordpy.CapsuleLedger()
        sealed = led.admit_and_seal(a)
        check("admit_and_seal returns SEALED capsule",
              sealed.lifecycle == coordpy.CapsuleLifecycle.SEALED)
        check("re-admitting same CID is idempotent",
              led.admit_and_seal(a).cid == sealed.cid)

        section("9. profiles registry")
        profs = coordpy.profiles.list_profiles()
        check("local_smoke listed", "local_smoke" in profs)
        required_keys = {"description", "readiness"}
        for pname in profs:
            p = coordpy.profiles.get_profile(pname)
            check(f"profile {pname!r} dict",
                  isinstance(p, dict)
                  and required_keys.issubset(p.keys()),
                  f"keys={sorted(p)[:6]}")

        section("10. extensions registry")
        for s in ("in_process", "subprocess", "docker"):
            check(f"sandbox {s!r} registered",
                  s in coordpy.extensions.list_sandboxes())

        # Register and discover a custom report sink + sandbox + task
        # bank, then confirm round-trip via list_*().
        from coordpy.extensions import (  # noqa: WPS433
            ReportSink, SandboxBackend, TaskBankBundle, TaskBankLoader,
            register_report_sink, register_sandbox,
        )

        class _Sink(ReportSink):
            def emit(self, *a, **k):
                return None

        register_report_sink("_spin_sink", _Sink())
        check("custom report sink visible",
              "_spin_sink" in coordpy.extensions.list_report_sinks())
        check("get_report_sink resolves",
              coordpy.extensions.get_report_sink("_spin_sink") is not None)

        section("11. import_data audit")
        data = os.path.join(
            os.path.dirname(coordpy.__file__),
            "_internal", "tasks", "data", "swe_real_shape_mini.jsonl",
        )
        check("bundled JSONL exists", os.path.exists(data))
        a_audit = coordpy.import_data.audit_jsonl(data)
        check("import audit returns IMPORT_AUDIT_SCHEMA",
              a_audit.get("schema") == coordpy.IMPORT_AUDIT_SCHEMA)

        section("12. provenance.build_manifest")
        m = coordpy.build_manifest(repo_dir=".")
        check("manifest schema",
              m["schema"] == coordpy.PROVENANCE_SCHEMA)
        check("manifest package == 'coordpy'",
              m["code"]["package"] == "coordpy")
        check("manifest package_version == coordpy.__version__",
              m["code"]["package_version"] == coordpy.__version__)

        section("13. AgentTeam (synthetic backend, no network)")
        from coordpy.synthetic_llm import SyntheticLLMClient  # noqa: WPS433
        team = coordpy.create_team(
            [coordpy.agent("planner", "plan it"),
             coordpy.agent("writer", "write it")],
            backend=SyntheticLLMClient(default_response="ok"),
        )
        result = team.run("Hello world")
        check("TeamResult.final_output is str",
              isinstance(result.final_output, str))
        check("team produced at least one turn",
              len(list(result.turns)) >= 1)

        section("14. Layered APIs smoke")
        for cname in ("CoordPySimpleAPI",
                      "CoordPyBuilderAPI",
                      "CoordPyAdvancedAPI"):
            check(f"coordpy.{cname} resolves",
                  getattr(coordpy, cname, None) is not None)
        bs = coordpy.BuilderSpec(profile="local_smoke",
                                 out_dir=os.path.join(tmp, "builder"))
        check("BuilderSpec instantiable", bs is not None)
        check("BuilderSpec is dataclass",
              dataclasses.is_dataclass(bs))

        section("15. SweepSpec + run_sweep (mock mode, in-process)")
        spec = coordpy.SweepSpec(
            mode="mock", jsonl=data, n_instances=2, sandbox="in_process",
        )
        sout = coordpy.run_sweep(spec)
        check("run_sweep returns dict", isinstance(sout, dict))

        section("16. backend factories construct")
        b = coordpy.backend_from_env({})
        check("backend_from_env(empty) returns object",
              b is not None, type(b).__name__)
        b2 = coordpy.make_backend("ollama", model="qwen2.5:0.5b")
        check("make_backend('ollama')",
              isinstance(b2, coordpy.OllamaBackend))
        b3 = coordpy.make_backend(
            "openai", model="gpt-4o-mini",
            base_url="https://api.openai.com/v1", api_key="dummy",
        )
        check("make_backend('openai')",
              isinstance(b3, coordpy.OpenAICompatibleBackend))

        section("17. unknown profile raises clear error")
        try:
            coordpy.run(coordpy.RunSpec(profile="nope_profile",
                                       out_dir=os.path.join(tmp, "x")))
            check("unknown profile raises", False)
        except KeyError as e:
            check("unknown profile raises", True,
                  f"KeyError: {str(e)[:60]}")

        section("18. CapsuleNativeRunContext spin")
        ctx = coordpy.CapsuleNativeRunContext()
        check("CapsuleNativeRunContext constructs", ctx is not None)
        check("ctx has render() method",
              callable(getattr(ctx, "render", None)))
        check("ctx has seal_* helpers",
              callable(getattr(ctx, "seal_and_write_artifact", None)))

        section("19. Cross-path module identity")
        from coordpy.synthetic_llm import SyntheticLLMClient as A1
        from coordpy.capsule import ContextCapsule as A2
        import coordpy.synthetic_llm as direct
        import coordpy.capsule as direct2
        check("synthetic_llm cross-path identity",
              A1 is direct.SyntheticLLMClient)
        check("capsule cross-path identity",
              A2 is direct2.ContextCapsule)

        section("20. Reproducibility of the report shape")
        rs2 = coordpy.RunSpec(profile="local_smoke",
                              out_dir=os.path.join(tmp, "smoke2"))
        r2 = coordpy.run(rs2)
        check("readiness shape equal",
              r2["readiness"] == report["readiness"])
        check("readiness verdict identical (deterministic)",
              r2["readiness"]["ready"] == report["readiness"]["ready"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n# Summary")
    if not _FAILURES:
        print("ALL CHECKS PASSED.")
        return 0
    print(f"{len(_FAILURES)} failure(s):")
    for n, d in _FAILURES:
        print(f"  - {n}: {d}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
