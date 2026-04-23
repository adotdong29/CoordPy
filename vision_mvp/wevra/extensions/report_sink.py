"""``ReportSink`` extension point.

A ReportSink consumes a ``product_report.json``-shape dict — after a
run completes — and does something with it: emit to a Slack webhook,
POST to a dashboard, write a secondary artifact, push a Prometheus
gauge, etc.

Wevra ships two built-ins:

  * ``stdout``  — prints ``product_summary.txt`` to stdout
  * ``jsonfile``— writes the full report to a configurable path

External packages can register new sinks via the registry or
``entry_points = "wevra.report_sinks"``. The runner calls every
registered sink that the ``RunSpec.report_sinks`` list names, in
order, after the standard artifact set has landed.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ReportSink(Protocol):

    def name(self) -> str: ...
    def emit(self, report: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Handle ``report``. Return a small dict for the audit trail
        (e.g. ``{"ok": True, "wrote": "/tmp/report.json"}``).
        """
        ...


ReportSinkFactory = Callable[..., ReportSink]


_REGISTRY: dict[str, ReportSinkFactory] = {}


def register_report_sink(name: str, factory: ReportSinkFactory,
                          *, overwrite: bool = False) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("report-sink name must be a non-empty string")
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"report sink {name!r} is already registered; "
            f"pass overwrite=True to replace")
    _REGISTRY[name] = factory


def get_report_sink(name: str, **kwargs: Any) -> ReportSink:
    _ensure_builtins_registered()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown report sink {name!r}; "
            f"known: {sorted(_REGISTRY)}")
    inst = _REGISTRY[name](**kwargs)
    if not isinstance(inst, ReportSink):
        raise TypeError(
            f"factory for {name!r} returned {type(inst).__name__}, "
            f"which does not satisfy ReportSink")
    return inst


def list_report_sinks() -> list[str]:
    _ensure_builtins_registered()
    return sorted(_REGISTRY)


def reset_report_sink_registry() -> None:
    _REGISTRY.clear()


def _ensure_builtins_registered() -> None:
    if "stdout" in _REGISTRY:
        return

    class _StdoutSink:
        def name(self) -> str:
            return "stdout"

        def emit(self, report: dict[str, Any], **_kw: Any) -> dict[str, Any]:
            text = report.get("summary_text") or ""
            if text:
                print(text)
            return {"ok": True, "chars_written": len(text)}

    class _JsonFileSink:
        def __init__(self, path: str | None = None) -> None:
            self._path = path

        def name(self) -> str:
            return "jsonfile"

        def emit(self, report: dict[str, Any], *,
                  path: str | None = None, **_kw: Any) -> dict[str, Any]:
            target = path or self._path
            if not target:
                raise ValueError(
                    "jsonfile sink requires path=... either at "
                    "construction or at emit time")
            os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
            with open(target, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, default=str)
            return {"ok": True, "wrote": os.path.abspath(target)}

    _REGISTRY["stdout"] = lambda **_kw: _StdoutSink()
    _REGISTRY["jsonfile"] = lambda path=None, **_kw: _JsonFileSink(path=path)


__all__ = [
    "ReportSink", "ReportSinkFactory",
    "register_report_sink", "get_report_sink", "list_report_sinks",
    "reset_report_sink_registry",
]
