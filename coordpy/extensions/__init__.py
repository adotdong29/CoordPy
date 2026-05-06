"""CoordPy extension system.

Three extension points, each a runtime-checkable Protocol:

  * ``SandboxBackend``   — where patched code is executed
  * ``TaskBankLoader``   — where evaluation inputs come from
  * ``ReportSink``       — where ``product_report.json`` is sent

Each extension point has:

  1. a typed Protocol in the relevant submodule;
  2. a built-in registry (in-process ``register_*`` / ``get_*``);
  3. ``entry_points`` discovery under the groups
     ``"coordpy.sandboxes"``, ``"coordpy.task_banks"``,
     ``"coordpy.report_sinks"``.

An external package can add an extension by either
``coordpy.extensions.register_sandbox("my_box", MySandbox)`` at import
time, or by declaring an ``entry_points`` block in its own
``pyproject.toml`` that CoordPy discovers via
``importlib.metadata.entry_points()``.

This surface is contract-tested
(``coordpy/tests/test_coordpy_extensions.py``). Any rename or
removal here is a breaking SDK change and must bump
``coordpy.SDK_VERSION``.
"""

from __future__ import annotations

from .sandbox import (
    SandboxBackend, get_sandbox, list_sandboxes,
    register_sandbox, reset_sandbox_registry,
)
from .taskbank import (
    TaskBankLoader, TaskBankBundle, get_task_bank, list_task_banks,
    register_task_bank, reset_task_bank_registry,
)
from .report_sink import (
    ReportSink, get_report_sink, list_report_sinks,
    register_report_sink, reset_report_sink_registry,
)
from .registry import discover_entry_points, all_extensions


__all__ = [
    # Sandbox
    "SandboxBackend", "get_sandbox", "list_sandboxes",
    "register_sandbox", "reset_sandbox_registry",
    # Task bank
    "TaskBankLoader", "TaskBankBundle",
    "get_task_bank", "list_task_banks",
    "register_task_bank", "reset_task_bank_registry",
    # Report sink
    "ReportSink", "get_report_sink", "list_report_sinks",
    "register_report_sink", "reset_report_sink_registry",
    # Registry meta
    "discover_entry_points", "all_extensions",
]
