"""Phase-45 product report renderer.

Turns a ``product_report.json``-shape dict into a short human-
readable summary. Kept separate from the runner so the renderer
can be reused by tests + CI-gate scripts on a stored artifact.
"""

from __future__ import annotations

from typing import Any


def render_summary(report: dict[str, Any]) -> str:
    lines: list[str] = []
    prof = report.get("profile", "<unknown>")
    desc = report.get("profile_description", "")
    lines.append(f"=== context-zero product report ===")
    lines.append(f"profile     : {prof}")
    lines.append(f"description : {desc}")
    lines.append(f"out_dir     : {report.get('out_dir', '-')}")
    lines.append(f"wall_seconds: {report.get('wall_seconds', '-')}")
    lines.append("")

    rd = report.get("readiness") or {}
    ready = rd.get("ready", False)
    lines.append(f"readiness   : {'READY' if ready else 'NOT READY'}  "
                  f"(n={rd.get('n','-')}, passed={rd.get('n_passed_all','-')}, "
                  f"wall={rd.get('wall_seconds','-')}s)")
    for (name, r) in (rd.get("checks") or {}).items():
        lines.append(f"  - {name:<12s} passed={r.get('passed',0):>3}  "
                      f"failed={r.get('failed',0):>3}")
    blockers = rd.get("blockers") or []
    if blockers:
        lines.append(f"  blockers  : {blockers}")
    lines.append("")

    sw = report.get("sweep")
    if sw is None:
        lines.append("sweep       : not configured in this profile")
    elif sw.get("skipped"):
        lines.append(f"sweep       : SKIPPED ({sw.get('reason','?')})")
    elif sw.get("mode") == "mock":
        lines.append(f"sweep       : mock oracle, {len(sw.get('cells',[]))} "
                      f"cell(s), sandbox={sw.get('sandbox','-')}")
        for c in sw.get("cells", []):
            pooled = c.get("pooled", {})
            row = []
            for strat, p in pooled.items():
                row.append(
                    f"{strat}={p.get('pass_at_1', 0):.3f}")
            lines.append(f"  - parser={c['parser_mode']:<7s} "
                          f"apply={c['apply_mode']:<7s} "
                          f"nd={c['n_distractors']:<2d} "
                          f"n={c.get('n_instances', '?')} | "
                          + " ".join(row))
    else:
        lines.append("sweep       : real-LLM launch recorded "
                      "(operator-run on ASPEN cluster)")
        lines.append(f"  launch_cmd: {' '.join(sw.get('launch_cmd') or [])}")
        if sw.get("raw_capture_launch_cmd"):
            lines.append(f"  capture   : "
                          f"{' '.join(sw.get('raw_capture_launch_cmd') or [])}")
        lines.append(f"  note      : {sw.get('note','')}")
    lines.append("")
    lines.append(f"artifacts   : {report.get('artifacts', [])}")
    return "\n".join(lines) + "\n"
