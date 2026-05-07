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
    elif sw.get("executed_in_process") and sw.get("cells") is not None:
        mode = sw.get("mode", "?")
        lines.append(f"sweep       : {mode} EXECUTED in-process, "
                      f"{len(sw.get('cells',[]))} cell(s), "
                      f"sandbox={sw.get('sandbox','-')}"
                      + (f", model={sw.get('model')}" if sw.get("model") else ""))
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
    elif sw.get("requires_acknowledgement"):
        lines.append("sweep       : real-LLM launch STAGED "
                      "(acknowledge_heavy=False)")
        lines.append(f"  model     : {sw.get('model')}")
        lines.append(f"  endpoint  : {sw.get('endpoint')}")
        lines.append(f"  launch_cmd: {' '.join(sw.get('launch_cmd') or [])}")
        if sw.get("note"):
            lines.append(f"  note      : {sw['note']}")
    elif sw.get("error_kind"):
        lines.append(f"sweep       : ERROR {sw['error_kind']}: "
                      f"{sw.get('error_detail','')}")
    else:
        # Legacy v1 stub shape (mode='real', no executed_in_process).
        lines.append("sweep       : real-LLM launch recorded "
                      "(legacy staging mode)")
        lines.append(f"  launch_cmd: {' '.join(sw.get('launch_cmd') or [])}")
    lines.append("")
    cv = report.get("capsules")
    if isinstance(cv, dict) and cv.get("schema") == "coordpy.capsule_view.v1":
        stats = cv.get("stats", {})
        by_kind = stats.get("by_kind", {})
        chain_ok = bool(cv.get("chain_ok"))
        root = (cv.get("root_cid") or "")[:16]
        lines.append(
            f"capsules    : n={stats.get('n_entries', '-')}  "
            f"sealed={stats.get('n_sealed', '-')}  "
            f"chain_ok={'yes' if chain_ok else 'NO'}  "
            f"root={root}…")
        if by_kind:
            kind_str = ", ".join(
                f"{k}={v}" for k, v in sorted(by_kind.items()))
            lines.append(f"  by_kind   : {kind_str}")
    lines.append(f"artifacts   : {report.get('artifacts', [])}")
    return "\n".join(lines) + "\n"


__all__ = ("render_summary",)


def __dir__() -> list[str]:
    return sorted(__all__)
