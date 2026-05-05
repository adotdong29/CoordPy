"""``SandboxBackend`` extension point.

A SandboxBackend is the boundary where patched code + hidden tests
run. CoordPy ships three built-ins (``in_process``, ``subprocess``,
``docker``) wired to the existing ``vision_mvp.tasks.swe_sandbox``
implementations. External packages can register new backends via
the in-process registry or an ``entry_points`` block
(``coordpy.sandboxes``).

The Protocol is deliberately narrow: it names a backend, reports
availability, and runs one (patch, test_source) pair. This is the
smallest interface that lets a new backend (e.g. a remote executor,
a Firecracker microVM, a WASM runtime) slot in without touching the
runner.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class SandboxBackend(Protocol):
    """Narrow execution-boundary protocol.

    The concrete object returned by a factory must expose ``name()``,
    ``is_available()``, and ``run(...)``. Its ``run`` signature is the
    ``vision_mvp.tasks.swe_sandbox.Sandbox`` protocol, so the existing
    ``in_process`` / ``subprocess`` / ``docker`` backends satisfy this
    interface by construction.
    """

    def name(self) -> str: ...
    def is_available(self) -> bool: ...
    def run(self, **kwargs: Any) -> Any: ...


# A "factory" produces a concrete SandboxBackend instance. We keep
# the factory indirection so extensions can receive per-invocation
# kwargs (e.g. a custom docker image).
SandboxFactory = Callable[..., SandboxBackend]


_REGISTRY: dict[str, SandboxFactory] = {}


def register_sandbox(name: str, factory: SandboxFactory,
                      *, overwrite: bool = False) -> None:
    """Register a sandbox backend under ``name``.

    Raises ``ValueError`` on conflict unless ``overwrite=True``. The
    factory is called with no positional args and arbitrary kwargs at
    ``get_sandbox`` time; for zero-config backends, a class object is
    a valid factory.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("sandbox name must be a non-empty string")
    if name in _REGISTRY and not overwrite:
        raise ValueError(
            f"sandbox {name!r} is already registered; "
            f"pass overwrite=True to replace")
    _REGISTRY[name] = factory


def get_sandbox(name: str, **kwargs: Any) -> SandboxBackend:
    """Instantiate a sandbox backend by name.

    Built-ins (``in_process`` / ``subprocess`` / ``docker``) are
    lazily registered on first call so the extension package has no
    import-order dependency on ``vision_mvp.tasks``.
    """
    _ensure_builtins_registered()
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown sandbox {name!r}; known: {sorted(_REGISTRY)}")
    inst = _REGISTRY[name](**kwargs)
    if not isinstance(inst, SandboxBackend):
        raise TypeError(
            f"factory for {name!r} returned {type(inst).__name__}, "
            f"which does not satisfy SandboxBackend")
    return inst


def list_sandboxes() -> list[str]:
    _ensure_builtins_registered()
    return sorted(_REGISTRY)


def reset_sandbox_registry() -> None:
    """Test-only: clear the registry (re-populated lazily)."""
    _REGISTRY.clear()


def _ensure_builtins_registered() -> None:
    if "in_process" in _REGISTRY:
        return
    from coordpy._internal.tasks.swe_sandbox import (
        InProcessSandbox, SubprocessSandbox, DockerSandbox,
    )
    _REGISTRY["in_process"] = lambda **_kw: InProcessSandbox()
    _REGISTRY["subprocess"] = lambda **_kw: SubprocessSandbox()
    _REGISTRY["docker"] = lambda image="python:3.11-slim", **_kw: (
        DockerSandbox(image=image))


__all__ = [
    "SandboxBackend", "SandboxFactory",
    "register_sandbox", "get_sandbox", "list_sandboxes",
    "reset_sandbox_registry",
]
