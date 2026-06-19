"""Example ReportSink: write the full report as JSON to a file.

This is a slightly richer variant of the built-in ``jsonfile`` sink
— it also writes a small ``.meta.json`` sidecar with the schema
constants so downstream tooling can confirm the shape without
reading the full report.

Used by ``test_coordpy_extensions.py`` to verify the end-to-end
extension path: register → runner.emit_sinks → file on disk.

Drop into an external package like this::

    # my_pkg/sinks.py
    from coordpy.extensions.examples.jsonl_report_sink import (
        JsonlWithMetaSink,
    )

    # pyproject.toml
    # [project.entry-points."coordpy.report_sinks"]
    # jsonl_with_meta = "my_pkg.sinks:JsonlWithMetaSink"
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class JsonlWithMetaSink:
    path: str | None = None

    def name(self) -> str:
        return "jsonl_with_meta"

    def emit(self, report: dict[str, Any], *,
              path: str | None = None,
              **_kw: Any) -> dict[str, Any]:
        target = path or self.path
        if not target:
            raise ValueError(
                "JsonlWithMetaSink requires path=... either at "
                "construction or at emit time")
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        meta_path = target + ".meta.json"
        meta = {
            "schema": report.get("schema"),
            "profile": report.get("profile"),
            "provenance_schema": (
                report.get("provenance", {}).get("schema")),
            "artifact_count": len(report.get("artifacts", [])),
        }
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return {"ok": True,
                "wrote": os.path.abspath(target),
                "meta": os.path.abspath(meta_path)}


__all__ = ["JsonlWithMetaSink"]
