"""Extension-discovery registry meta-layer.

Three entry-point groups are searched:

  * ``wevra.sandboxes``    → ``register_sandbox(name, factory)``
  * ``wevra.task_banks``   → ``register_task_bank(name, factory)``
  * ``wevra.report_sinks`` → ``register_report_sink(name, factory)``

Each entry point is expected to load to a callable (the factory)
producing an object that satisfies the relevant Protocol. Loader
errors are collected and returned so operators can see *why* a
third-party plugin didn't load, rather than silently failing.
"""

from __future__ import annotations

import importlib.metadata as _im
from typing import Any

from .sandbox import register_sandbox, list_sandboxes
from .taskbank import register_task_bank, list_task_banks
from .report_sink import register_report_sink, list_report_sinks


_DISCOVERY_GROUPS = {
    "wevra.sandboxes": register_sandbox,
    "wevra.task_banks": register_task_bank,
    "wevra.report_sinks": register_report_sink,
}


def discover_entry_points(*, overwrite: bool = False) -> dict[str, Any]:
    """Scan installed packages for Wevra extension entry points.

    Returns a summary dict::

        {
          "discovered": {"wevra.sandboxes": ["my_firecracker", ...], ...},
          "errors":     [{"group": "...", "name": "...", "error": "..."}],
        }

    Safe to call multiple times. ``overwrite=False`` means a plugin
    that collides with a built-in raises an error captured in
    ``errors``, not a hard crash.
    """
    discovered: dict[str, list[str]] = {g: [] for g in _DISCOVERY_GROUPS}
    errors: list[dict[str, str]] = []

    try:
        all_eps = _im.entry_points()
    except Exception as ex:  # pragma: no cover
        return {"discovered": discovered,
                "errors": [{"group": "<meta>", "name": "<meta>",
                             "error": f"entry_points() failed: {ex}"}]}

    for group, register_fn in _DISCOVERY_GROUPS.items():
        try:
            eps = all_eps.select(group=group)  # type: ignore[attr-defined]
        except AttributeError:
            # Older API: dict-shaped.
            eps = all_eps.get(group, [])
        for ep in eps:
            try:
                factory = ep.load()
                register_fn(ep.name, factory, overwrite=overwrite)
                discovered[group].append(ep.name)
            except Exception as ex:
                errors.append({
                    "group": group,
                    "name": getattr(ep, "name", "<?>"),
                    "error": f"{type(ex).__name__}: {ex}",
                })
    return {"discovered": discovered, "errors": errors}


def all_extensions() -> dict[str, list[str]]:
    """Return {group: [name, ...]} of every currently registered
    extension (built-in + third-party)."""
    return {
        "wevra.sandboxes": list_sandboxes(),
        "wevra.task_banks": list_task_banks(),
        "wevra.report_sinks": list_report_sinks(),
    }


__all__ = ["discover_entry_points", "all_extensions"]
