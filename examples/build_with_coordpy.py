"""End-to-end demo of the coordpy SDK.

Runs without a network or an LLM by using the synthetic backend, so
anyone can execute it after a fresh install:

    pip install coordpy-ai
    python examples/build_with_coordpy.py

The demo walks through eight steps:

    1. Spin up a three-agent team with the synthetic backend.
    2. Run the bundled ``local_smoke`` profile and produce a RunReport.
    3. Re-verify the on-disk capsule chain from bytes.
    4. Run the lifecycle audit.
    5. Apply the ``coordpy-ci`` gate programmatically.
    6. Register a custom ReportSink and emit through it.
    7. Build a skeleton ledger from the finished report.
    8. Print a green/red summary.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import coordpy
from coordpy.extensions import register_report_sink, ReportSink
from coordpy.synthetic_llm import SyntheticLLMClient


def _heading(n: int, title: str) -> None:
    bar = "─" * 60
    print(f"\n{bar}\n  {n}. {title}\n{bar}")


def main() -> int:
    workdir = tempfile.mkdtemp(prefix="coordpy_demo_")
    try:
        _heading(1, "Spin up a 3-agent team (synthetic backend)")
        team = coordpy.create_team(
            [
                coordpy.agent(
                    "planner",
                    "Break the user task into 2-3 crisp steps.",
                ),
                coordpy.agent(
                    "researcher",
                    "Gather the facts that matter for each step.",
                ),
                coordpy.agent(
                    "writer",
                    "Write the final answer for the user.",
                ),
            ],
            backend=SyntheticLLMClient(
                default_response=(
                    "step 1: list the bounded-context idea; "
                    "step 2: cite capsules; step 3: write."
                ),
            ),
            team_instructions=(
                "Reuse visible handoffs instead of restating "
                "the full task each turn."
            ),
        )
        result = team.run("Explain CoordPy in one paragraph.")
        print(f"agents       : {[a.name for a in team.agents]}")
        print(f"turns        : {len(list(result.turns))}")
        print(f"final_output : {result.final_output[:80]}…")
        if result.root_cid:
            print(f"root_cid     : {result.root_cid[:16]}…")

        _heading(2, "Run the local_smoke profile → RunReport")
        out_dir = os.path.join(workdir, "smoke")
        rs = coordpy.RunSpec(profile="local_smoke", out_dir=out_dir)
        report = coordpy.run(rs)
        print(f"profile      : {rs.profile}")
        print(f"ready        : {report['readiness']['ready']}")
        print(f"sdk          : {coordpy.SDK_VERSION}")
        print(f"package_ver  : {report['provenance']['code']['package_version']}")
        print(f"capsules     : {report['capsules']['stats']['n_entries']} "
              f"(chain_ok={report['capsules']['chain_ok']})")
        print(f"root_cid     : {report['capsules']['root_cid'][:16]}…")

        _heading(3, "Re-verify the capsule chain from on-disk bytes")
        view_path = os.path.join(out_dir, "capsule_view.json")
        with open(view_path, encoding="utf-8") as fh:
            disk_view = json.load(fh)
        ok_disk = coordpy.verify_chain_from_view_dict(disk_view)
        ac = coordpy.verify_artifacts_on_disk(disk_view, base_dir=out_dir)
        mm_path = os.path.join(out_dir, "meta_manifest.json")
        if os.path.exists(mm_path):
            with open(mm_path, encoding="utf-8") as fh:
                mm = json.load(fh)
            mc = coordpy.verify_meta_manifest_on_disk(mm, base_dir=out_dir)
            mm_verdict = mc["verdict"]
        else:
            mm_verdict = "ABSENT"
        print(f"chain_ok_on_disk      : {ok_disk}")
        print(f"artifacts_match       : {ac['verdict']} "
              f"({ac['ok']}/{ac['checked']})")
        print(f"meta_manifest_match   : {mm_verdict}")

        _heading(4, "Lifecycle audit (T-1..T-7 invariants)")
        audit = coordpy.audit_capsule_lifecycle_from_view(report["capsules"])
        print(f"verdict      : {audit.verdict}")
        print(f"violations   : {len(getattr(audit, 'violations', []))}")

        _heading(5, "Apply the CI gate programmatically")
        report_path = os.path.join(out_dir, "product_report.json")
        verdict = coordpy.ci_gate.evaluate_report(
            report_path, min_pass_at_1=1.0,
        )
        print(f"ok           : {verdict['ok']}")
        for cname, sub in verdict["checks"].items():
            print(f"  - {cname:<11s} : {sub.get('ok')}")

        _heading(6, "Register a custom ReportSink and emit through it")

        class CountingSink(ReportSink):
            def __init__(self) -> None:
                self.calls = 0

            def name(self) -> str:
                return "demo_counter"

            def emit(self, report, **kwargs):
                self.calls += 1
                return {"ok": True, "calls": self.calls}

        sink = CountingSink()
        register_report_sink("demo_counter", sink, overwrite=True)
        out = sink.emit(report)
        print(f"registered  : {'demo_counter' in coordpy.extensions.list_report_sinks()}")
        print(f"sink_emit    : {out}")

        _heading(7, "Skeleton ledger from RunReport (build_report_ledger)")
        # Folds the finished report into a high-level capsule skeleton
        # (PROFILE / READINESS / SWEEP_SPEC / SWEEP_CELL / PROVENANCE /
        # ARTIFACT / RUN_REPORT) — useful for downstream tools that
        # only care about run-shape, not the per-instance capsules.
        led, root = coordpy.build_report_ledger(report)
        v = coordpy.render_view(led, root_cid=root)
        print(f"skeleton n   : {len(v.capsules)}")
        print(f"chain_ok     : {v.chain_ok}")
        print(f"skel root    : {root[:16]}…")
        print(f"deep root    : {report['capsules']['root_cid'][:16]}…  "
              f"(154-capsule deep graph)")

        _heading(8, "Summary")
        all_green = (
            report["readiness"]["ready"]
            and report["capsules"]["chain_ok"]
            and ok_disk
            and ac["verdict"] in ("OK", "EMPTY")
            and mm_verdict in ("OK", "ABSENT", "EMPTY")
            and audit.verdict == "OK"
            and verdict["ok"]
            and v.chain_ok
        )
        print(f"  all gates green : {all_green}")
        print(f"  artefacts in   : {out_dir}")
        return 0 if all_green else 1
    finally:
        # Tidy up unless the user asked us to keep it.
        if not os.environ.get("COORDPY_KEEP_DEMO_DIR"):
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
