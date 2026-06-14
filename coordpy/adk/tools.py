"""coordpy.adk.tools — tools are typed, docstring'd Python functions.

You pass plain functions into ``Agent(tools=[...])``; ADK wraps each one in
a ``FunctionTool`` automatically. The signature *is* the schema the model
sees: parameter names + type hints + the docstring. A trailing parameter
named ``tool_context`` (or annotated ``ToolContext``) is recognised and
auto-injected at call time — it never appears in the model-facing schema.

By convention a tool returns a ``dict`` with a ``status`` key
(``"success"`` / ``"error"``); a non-dict return is wrapped as
``{"result": value}`` so the trail stays structured.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable


def _type_name(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return "Any"
    name = getattr(annotation, "__name__", None)
    if name:
        return name
    return str(annotation).replace("typing.", "")


def _is_tool_context_param(p: inspect.Parameter) -> bool:
    if p.name == "tool_context":
        return True
    ann = p.annotation
    ann_name = getattr(ann, "__name__", "") or str(ann)
    return ann_name.endswith("ToolContext")


class BaseTool:
    """Minimal tool interface. ``name``/``description`` feed the prompt."""

    name: str = "tool"
    description: str = ""

    def params_sig(self) -> str:
        return ""

    def run(self, args: dict[str, Any], tool_context: Any) -> Any:
        raise NotImplementedError


class FunctionTool(BaseTool):
    """Wrap a Python function as a tool, deriving its schema by reflection."""

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        self.name = getattr(func, "__name__", "tool")
        self.description = (inspect.getdoc(func) or "").strip().split("\n\n")[0]
        self._params: list[tuple[str, str, bool]] = []
        self._wants_ctx = False
        self._ctx_param_name = "tool_context"
        for p in inspect.signature(func).parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
                continue
            if _is_tool_context_param(p):
                self._wants_ctx = True
                self._ctx_param_name = p.name
                continue
            required = p.default is inspect.Parameter.empty
            self._params.append((p.name, _type_name(p.annotation), required))

    @property
    def param_names(self) -> set[str]:
        return {n for (n, _t, _r) in self._params}

    def params_sig(self) -> str:
        return ", ".join(f"{n}: {t}" for (n, t, _r) in self._params)

    def run(self, args: dict[str, Any], tool_context: Any) -> Any:
        kwargs: dict[str, Any] = {
            k: v for k, v in (args or {}).items() if k in self.param_names}
        if self._wants_ctx:
            kwargs[self._ctx_param_name] = tool_context
        result = self.func(**kwargs)
        if not isinstance(result, dict):
            return {"result": result}
        return result


def to_function_tool(obj: Any) -> BaseTool:
    """Coerce a function (or pass through a BaseTool) into a tool."""
    if isinstance(obj, BaseTool):
        return obj
    if callable(obj):
        return FunctionTool(obj)
    raise TypeError(f"tools must be callables or BaseTool, got {type(obj)!r}")
