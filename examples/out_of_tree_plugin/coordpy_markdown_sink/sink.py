"""Minimal ReportSink that renders a CoordPy product report as Markdown.

This module is the entry-point target declared in the package
``pyproject.toml``:

    [project.entry-points."coordpy.report_sinks"]
    markdown = "coordpy_markdown_sink.sink:register"

When CoordPy's extension registry first resolves report sinks, it calls
``register()`` below, which asks CoordPy to register a factory for a
``ReportSink`` named ``"markdown"``. After that, the sink is usable
from the CoordPy CLI (``coordpy --report-sink markdown``) and from
``RunSpec(report_sinks=("markdown",))``.
"""

from __future__ import annotations

import os
from typing import Any


class MarkdownReportSink:
    """Render a CoordPy product report as a short Markdown digest."""

    def __init__(self, path: str | None = None) -> None:
        # ``path`` may be set at construction OR at emit time. If
        # neither is provided, the sink writes ``report.md`` next to
        # the run's ``product_report.json``.
        self._path = path

    def name(self) -> str:
        return "markdown"

    def emit(self, report: dict[str, Any], *,
             path: str | None = None, **_kw: Any) -> dict[str, Any]:
        target = path or self._path or self._default_path(report)
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            fh.write(self._render(report))
        return {"ok": True, "wrote": os.path.abspath(target),
                "format": "markdown"}

    @staticmethod
    def _default_path(report: dict[str, Any]) -> str:
        out_dir = (report.get("run", {}) or {}).get("out_dir") or "."
        return os.path.join(out_dir, "report.md")

    @staticmethod
    def _render(report: dict[str, Any]) -> str:
        readiness = report.get("readiness", {}) or {}
        sweep = report.get("sweep", {}) or {}
        prov = report.get("provenance", {}) or {}
        lines = [
            "# CoordPy run report",
            "",
            f"- **Profile:** `{report.get('profile', '?')}`",
            f"- **Ready:** `{readiness.get('ready', '?')}`",
            f"- **Schema:** `{report.get('schema', '?')}`",
            f"- **Provenance schema:** `{prov.get('schema', '?')}`",
            f"- **Git SHA:** `{prov.get('git_sha', '?')}`",
            "",
            "## Sweep",
            "",
            f"- executed_in_process: `{sweep.get('executed_in_process', False)}`",
            f"- model: `{sweep.get('model', '?')}`",
            f"- instances: `{sweep.get('n_instances', '?')}`",
            "",
            "## Summary",
            "",
            "```",
            (report.get("summary_text") or "").rstrip() or "(no summary)",
            "```",
            "",
        ]
        return "\n".join(lines)


def register() -> None:
    """Register the markdown sink factory with CoordPy.

    Imported lazily so that simply importing this package does not
    force a transitive import of ``vision_mvp.coordpy``. CoordPy calls
    ``register()`` once, when it first resolves report-sink entry
    points.
    """
    from vision_mvp.coordpy.extensions import register_report_sink

    register_report_sink(
        "markdown",
        lambda path=None, **_kw: MarkdownReportSink(path=path),
        overwrite=True,
    )
