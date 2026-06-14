"""coordpy.adk.agents — Agent, LlmAgent, and the workflow agents.

``Agent`` (alias ``LlmAgent``) is the LLM-powered worker: give it a
``name``, a ``model`` (any CoordPy ``LLMBackend`` / callable / model tag),
an ``instruction``, ``tools``, optional ``sub_agents``, and an optional
``output_key``. ``agent.run(ctx)`` yields a *stream* of ``Event``s — tool
calls, transfers, and a final answer.

The reference tool-calling protocol is text-based so it works with ANY
backend (including the hermetic ``SyntheticLLMClient``): the model replies
with a single ``TOOL_CALL: {...}`` or ``TRANSFER: <name>`` line, or its
final answer.

The deterministic workflow agents — ``SequentialAgent`` / ``ParallelAgent``
/ ``LoopAgent`` — orchestrate ``sub_agents`` over shared ``State`` via
``output_key`` (no LLM routing).
"""

from __future__ import annotations

import dataclasses
import json
import re
import uuid
from typing import Any, Callable, Iterator, Sequence

from .context import CallbackContext, InvocationContext, ToolContext
from .sessions import Event, EventActions
from .tools import BaseTool, to_function_tool

_FIELD_RE = re.compile(r"\{([a-zA-Z_][\w:]*)\}")


def _template(text: str, state: Any) -> str:
    """Substitute ``{key}`` against state; leave unknown keys untouched."""
    if not text or "{" not in text:
        return text

    def repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        return str(state.get(key)) if key in state else m.group(0)

    return _FIELD_RE.sub(repl, text)


@dataclasses.dataclass
class _Parsed:
    kind: str  # "tool" | "transfer" | "final"
    text: str = ""
    tool_name: str | None = None
    args: dict[str, Any] = dataclasses.field(default_factory=dict)
    target: str | None = None


def _parse_response(text: str) -> _Parsed:
    for line in (text or "").splitlines():
        s = line.strip()
        if s.startswith("TOOL_CALL:"):
            try:
                obj = json.loads(s[len("TOOL_CALL:"):].strip())
            except Exception:
                continue
            return _Parsed(kind="tool", tool_name=obj.get("tool"),
                           args=obj.get("args") or {})
        if s.startswith("TRANSFER:"):
            return _Parsed(kind="transfer", target=s[len("TRANSFER:"):].strip())
    return _Parsed(kind="final", text=(text or "").strip())


class _CallableBackend:
    """Adapt a plain ``(prompt) -> str`` callable to the LLMBackend shape."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn
        self.model = "callable"
        self.base_url = None

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        return self._fn(prompt)


class BaseAgent:
    """Common agent surface: a name, a description, and a child hierarchy."""

    def __init__(self, *, name: str, description: str = "",
                 sub_agents: Sequence["BaseAgent"] = ()) -> None:
        self.name = name
        self.description = description
        self.sub_agents: list[BaseAgent] = list(sub_agents)
        self.parent_agent: BaseAgent | None = None
        for child in self.sub_agents:
            child.parent_agent = self

    def find_agent(self, name: str) -> "BaseAgent | None":
        if self.name == name:
            return self
        for child in self.sub_agents:
            found = child.find_agent(name)
            if found is not None:
                return found
        return None

    def run(self, ctx: InvocationContext) -> Iterator[Event]:
        raise NotImplementedError


class Agent(BaseAgent):
    """An LLM-driven agent (Google ADK's ``LlmAgent``)."""

    def __init__(self, *, name: str, model: Any = None, instruction: str = "",
                 description: str = "", tools: Sequence[Any] = (),
                 sub_agents: Sequence[BaseAgent] = (),
                 output_key: str | None = None,
                 before_agent_callback: Callable | None = None,
                 after_agent_callback: Callable | None = None,
                 before_model_callback: Callable | None = None,
                 after_model_callback: Callable | None = None,
                 before_tool_callback: Callable | None = None,
                 after_tool_callback: Callable | None = None,
                 max_tool_iterations: int = 4,
                 temperature: float = 0.0, max_tokens: int = 512) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self.model = model
        self.instruction = instruction
        self.output_key = output_key
        self.tools: list[BaseTool] = [to_function_tool(t) for t in tools]
        self._tools_by_name = {t.name: t for t in self.tools}
        self.before_agent_callback = before_agent_callback
        self.after_agent_callback = after_agent_callback
        self.before_model_callback = before_model_callback
        self.after_model_callback = after_model_callback
        self.before_tool_callback = before_tool_callback
        self.after_tool_callback = after_tool_callback
        self.max_tool_iterations = max_tool_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

    # -- model resolution -------------------------------------------------
    def _resolve_backend(self) -> Any:
        m = self.model
        if m is None:
            from coordpy.llm_backend import backend_from_env
            return backend_from_env()
        if hasattr(m, "generate"):
            return m
        if isinstance(m, str):
            from coordpy.llm_backend import backend_from_env
            return backend_from_env(model=m)
        if callable(m):
            return _CallableBackend(m)
        raise TypeError(
            f"model must be an LLMBackend, a model-tag str, or a callable; "
            f"got {type(m)!r}")

    # -- prompt assembly --------------------------------------------------
    def _protocol_block(self) -> str:
        lines = ["You may use a tool, hand off to a teammate, or answer "
                 "directly."]
        if self.tools:
            lines.append('To use a tool reply with ONE line: '
                         'TOOL_CALL: {"tool": "<name>", "args": {<json>}}')
            lines.append("Tools:")
            for t in self.tools:
                lines.append(f"- {t.name}({t.params_sig()}): {t.description}")
        if self.sub_agents:
            lines.append("To hand off reply with ONE line: TRANSFER: <teammate>")
            lines.append("Teammates:")
            for a in self.sub_agents:
                lines.append(f"- {a.name}: {a.description}")
        lines.append("Otherwise reply with your final answer for the user.")
        return "\n".join(lines)

    def _build_prompt(self, ctx: InvocationContext,
                      observations: list[tuple[str, Any]]) -> str:
        parts: list[str] = []
        instr = _template(self.instruction, ctx.state)
        if instr:
            parts.append(instr)
        parts.append(f"User request:\n{ctx.user_content}")
        if self.tools or self.sub_agents:
            parts.append(self._protocol_block())
        for tool_name, result in observations:
            parts.append(f"Observation from {tool_name}: "
                         f"{json.dumps(result, default=str)}")
        return "\n\n".join(parts)

    # -- event construction ----------------------------------------------
    def _attach_artifacts(self, ctx: InvocationContext, ev: Event) -> None:
        if ctx._pending_artifacts:
            for (fn, ver, data, mime) in ctx._pending_artifacts:
                ev.actions.artifact_delta[fn] = ver
                ev._artifact_blobs.append((fn, ver, data, mime))
            ctx._pending_artifacts.clear()

    def _final_event(self, ctx: InvocationContext, text: str) -> Event:
        ev = Event(author=self.name, content=text, is_final=True,
                   invocation_id=ctx.invocation_id)
        ev.actions.state_delta = ctx.state.pop_delta()
        self._attach_artifacts(ctx, ev)
        return ev

    # -- the turn ---------------------------------------------------------
    def run(self, ctx: InvocationContext) -> Iterator[Event]:
        ctx.agent = self
        cb = CallbackContext(ctx, agent_name=self.name)

        if self.before_agent_callback is not None:
            override = self.before_agent_callback(cb)
            if override is not None:
                yield self._final_event(ctx, str(override))
                return

        backend = self._resolve_backend()
        observations: list[tuple[str, Any]] = []
        final_text: str | None = None
        transfer_target: str | None = None

        for iteration in range(self.max_tool_iterations + 1):
            prompt = self._build_prompt(ctx, observations)

            response: Any = None
            if self.before_model_callback is not None:
                response = self.before_model_callback(cb, prompt)
            if response is None:
                response = backend.generate(
                    prompt, max_tokens=self.max_tokens,
                    temperature=self.temperature)
            if self.after_model_callback is not None:
                rewritten = self.after_model_callback(cb, response)
                if rewritten is not None:
                    response = rewritten

            parsed = _parse_response(str(response))

            if parsed.kind == "tool" and iteration < self.max_tool_iterations:
                tcx = ToolContext(ctx, agent_name=self.name,
                                  function_call_id="fc-" + uuid.uuid4().hex[:8])
                tool = self._tools_by_name.get(parsed.tool_name or "")
                if tool is None:
                    result: Any = {"status": "error",
                                   "error": f"unknown tool {parsed.tool_name!r}"}
                else:
                    pre = (self.before_tool_callback(tool, parsed.args, tcx)
                           if self.before_tool_callback is not None else None)
                    result = pre if pre is not None else tool.run(parsed.args, tcx)
                    if self.after_tool_callback is not None:
                        post = self.after_tool_callback(
                            tool, parsed.args, tcx, result)
                        if post is not None:
                            result = post
                ev = Event(author=self.name, content=None,
                           invocation_id=ctx.invocation_id,
                           tool_call={"tool": parsed.tool_name,
                                      "args": parsed.args},
                           tool_result=result, actions=tcx.actions)
                ev.actions.state_delta = ctx.state.pop_delta()
                self._attach_artifacts(ctx, ev)
                yield ev
                observations.append((parsed.tool_name or "tool", result))
                if tcx.actions.escalate or tcx.actions.transfer_to_agent:
                    transfer_target = tcx.actions.transfer_to_agent
                    break
                continue

            if parsed.kind == "transfer":
                transfer_target = parsed.target
                ev = Event(author=self.name, content=None,
                           invocation_id=ctx.invocation_id,
                           actions=EventActions(transfer_to_agent=parsed.target))
                ev.actions.state_delta = ctx.state.pop_delta()
                yield ev
                break

            final_text = parsed.text
            break

        if transfer_target:
            target = self.find_agent(transfer_target)
            if target is None and ctx.root_agent is not None:
                target = ctx.root_agent.find_agent(transfer_target)
            if target is not None and target is not self:
                yield from target.run(ctx)
                return

        if final_text is None:
            final_text = "(no final response)"
        if self.after_agent_callback is not None:
            ov = self.after_agent_callback(cb)
            if ov is not None:
                final_text = str(ov)
        if self.output_key:
            ctx.state[self.output_key] = final_text
        yield self._final_event(ctx, final_text)


# Google ADK's ``Agent`` is an alias of ``LlmAgent``; expose both.
LlmAgent = Agent


class SequentialAgent(BaseAgent):
    """Run ``sub_agents`` in order; each step's ``output_key`` feeds the
    next via shared ``State``. Early-exits if any event escalates."""

    def run(self, ctx: InvocationContext) -> Iterator[Event]:
        for child in self.sub_agents:
            for ev in child.run(ctx):
                yield ev
                if ev.actions.escalate:
                    return


class ParallelAgent(BaseAgent):
    """Run ``sub_agents`` as independent branches over the shared state.

    The reference implementation runs branches in order (deterministic);
    give branches distinct ``output_key``s so they don't clobber each other.
    """

    def run(self, ctx: InvocationContext) -> Iterator[Event]:
        for child in self.sub_agents:
            ev_branch = child.run(ctx)
            for ev in ev_branch:
                ev.branch = child.name
                yield ev


class LoopAgent(BaseAgent):
    """Repeat ``sub_agents`` until ``max_iterations`` or an event sets
    ``actions.escalate=True`` (a critic approving, say)."""

    def __init__(self, *, name: str, description: str = "",
                 sub_agents: Sequence[BaseAgent] = (),
                 max_iterations: int = 3) -> None:
        super().__init__(name=name, description=description, sub_agents=sub_agents)
        self.max_iterations = max_iterations

    def run(self, ctx: InvocationContext) -> Iterator[Event]:
        for _ in range(self.max_iterations):
            for child in self.sub_agents:
                for ev in child.run(ctx):
                    yield ev
                    if ev.actions.escalate:
                        return
