"""``TaskBankLoader`` extension point.

A TaskBankLoader resolves a *source descriptor* (today: a JSONL path)
into the in-memory bundle the runner needs: a list of
``SWEBenchStyleTask`` and a ``repo_files`` dict keyed by file path.

CoordPy ships one built-in: ``jsonl`` (the existing
``load_jsonl_bank``). External packages can register new loaders
(HuggingFace datasets, a remote bank server, a per-commit fetch) via
the in-process registry or ``entry_points = "coordpy.task_banks"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class TaskBankBundle:
    """What a loader returns to the runner.

    ``tasks`` is any iterable of ``SWEBenchStyleTask`` (opaque to the
    SDK — the concrete class lives in the research package). The
    runner drives it via the standard bank iteration protocol.
    """

    tasks: Any
    repo_files: dict[str, str]
    source: str
    schema: str = "coordpy.task_bank.v1"


@runtime_checkable
class TaskBankLoader(Protocol):

    def name(self) -> str: ...
    def load(self, source: str, **kwargs: Any) -> TaskBankBundle: ...


TaskBankFactory = Callable[..., TaskBankLoader]


_REGISTRY: dict[str, TaskBankFactory] = {}


def register_task_bank(
    name: str,
    factory: "TaskBankFactory | TaskBankLoader",
    *, overwrite: bool = False,
) -> None:
    """Register a task bank loader under ``name``.

    Accepts either a factory callable (returning a
    ``TaskBankLoader``) or an already-built loader instance.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("task-bank name must be a non-empty string")
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"task bank {name!r} is already registered; "
            f"pass overwrite=True to replace")
    if not callable(factory):
        instance = factory
        if not isinstance(instance, TaskBankLoader):
            raise TypeError(
                f"register_task_bank({name!r}, ...) requires either "
                f"a factory callable or a TaskBankLoader instance; "
                f"got {type(instance).__name__}")
        factory = lambda **_kw: instance  # noqa: E731
    _REGISTRY[name] = factory  # type: ignore[assignment]


def get_task_bank(name: str, **kwargs: Any) -> TaskBankLoader:
    _ensure_builtins_registered()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown task bank {name!r}; "
            f"known: {sorted(_REGISTRY)}")
    inst = _REGISTRY[name](**kwargs)
    if not isinstance(inst, TaskBankLoader):
        raise TypeError(
            f"factory for {name!r} returned {type(inst).__name__}, "
            f"which does not satisfy TaskBankLoader")
    return inst


def list_task_banks() -> list[str]:
    _ensure_builtins_registered()
    return sorted(_REGISTRY)


def reset_task_bank_registry() -> None:
    _REGISTRY.clear()


def _ensure_builtins_registered() -> None:
    if "jsonl" in _REGISTRY:
        return

    class _JSONLLoader:
        def name(self) -> str:
            return "jsonl"

        def load(self, source: str, *,
                  hidden_event_log_factory: Any = None,
                  limit: int | None = None,
                  **_kw: Any) -> TaskBankBundle:
            from coordpy._internal.tasks.swe_bench_bridge import (
                build_synthetic_event_log, load_jsonl_bank,
            )
            factory = hidden_event_log_factory or (
                lambda t, _k=6: build_synthetic_event_log(t, _k))
            tasks, repo_files = load_jsonl_bank(
                source, hidden_event_log_factory=factory, limit=limit)
            return TaskBankBundle(
                tasks=tasks, repo_files=repo_files, source=source)

    _REGISTRY["jsonl"] = lambda **_kw: _JSONLLoader()


__all__ = [
    "TaskBankBundle", "TaskBankLoader", "TaskBankFactory",
    "register_task_bank", "get_task_bank", "list_task_banks",
    "reset_task_bank_registry",
]
