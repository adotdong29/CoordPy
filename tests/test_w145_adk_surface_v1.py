"""W145 unit tests — the library-first ``coordpy.adk`` ADK surface.

Fast $0 tests (no NIM, no network, no real model — a tiny scripted backend
stands in for the LLM). They lock the ADK surface that W145 lands:

  * the public surface imports and is additive (legacy ``coordpy.Agent`` /
    ``create_team`` are unchanged; ``coordpy.adk.Agent`` is a separate object);
  * Agent + Runner + Session: a turn is an Event stream gated on
    ``is_final_response()``;
  * function tools + auto-injected ``ToolContext`` (state writes, artifacts);
  * ``State`` prefix routing (``user:`` persists cross-session, ``temp:`` is
    dropped) and the ``output_key`` data bus;
  * artifacts (versioned save / load / list);
  * ``sub_agents`` transfer + ``SequentialAgent`` / ``ParallelAgent`` /
    ``LoopAgent`` composition;
  * callbacks short-circuit;
  * **the CoordPy guarantee underneath**: every run seals a hash-chained
    capsule trail that re-verifies from bytes and is tamper-evident, with a
    deterministic ``root_cid`` for identical inputs;
  * memory recall; and the bundled example app.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import coordpy  # noqa: E402
import coordpy.adk as adk  # noqa: E402
from coordpy.adk import (  # noqa: E402
    Agent, InMemoryRunner, InMemorySessionService, LoopAgent, ParallelAgent,
    Runner, SequentialAgent, ToolContext,
)
from coordpy.capsule import verify_chain_from_view_dict  # noqa: E402


class ScriptedBackend:
    """Hermetic LLMBackend stand-in: returns responses by call order."""

    def __init__(self, *responses: str) -> None:
        self.model = "scripted"
        self.base_url = None
        self._i = -1
        self._responses = list(responses) or [""]

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        self._i += 1
        return self._responses[min(self._i, len(self._responses) - 1)]


def _make_runner(agent, app_name="app", session_id="s", user_id="u"):
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    runner.session_service.create_session(
        app_name=app_name, user_id=user_id, session_id=session_id)
    return runner


# --- surface + back-compat ------------------------------------------------

def test_adk_surface_imports():
    for name in ("Agent", "LlmAgent", "BaseAgent", "SequentialAgent",
                 "ParallelAgent", "LoopAgent", "BaseTool", "FunctionTool",
                 "to_function_tool", "Runner", "InMemoryRunner", "Session",
                 "State", "Event", "EventActions", "BaseSessionService",
                 "InMemorySessionService", "Artifact", "BaseArtifactService",
                 "InMemoryArtifactService", "MemoryEntry", "BaseMemoryService",
                 "InMemoryMemoryService", "ReadonlyContext", "CallbackContext",
                 "ToolContext", "InvocationContext"):
        assert hasattr(adk, name), f"coordpy.adk missing {name}"
    assert adk.ADK_SURFACE_SCHEMA == "coordpy.adk.v1"
    assert adk.LlmAgent is adk.Agent


def test_back_compat_legacy_surface_unchanged():
    # The legacy product Agent keeps its exact dataclass fields.
    fields = {f.name for f in dataclasses.fields(coordpy.Agent)}
    assert fields == {"name", "instructions", "role", "backend",
                      "temperature", "max_tokens"}
    # The ADK Agent is a *separate* object (no rename, no collision).
    assert coordpy.adk.Agent is not coordpy.Agent
    for n in ("create_team", "AgentTeam", "agent", "replay_team_result",
              "presets", "TeamResult"):
        assert hasattr(coordpy, n), f"legacy surface lost {n}"


def test_adk_reexported_from_package():
    assert hasattr(coordpy, "adk")
    assert "adk" in coordpy.__all__
    from coordpy.adk import Agent as AdkAgent
    assert AdkAgent is coordpy.adk.Agent


# --- a turn is an event stream -------------------------------------------

def test_single_turn_final_response():
    agent = Agent(name="a", model=ScriptedBackend("Hello there."),
                  instruction="Be brief.")
    runner = _make_runner(agent)
    events = list(runner.run(user_id="u", session_id="s", new_message="hi"))
    finals = [e for e in events if e.is_final_response()]
    assert len(finals) == 1
    assert finals[0].text == "Hello there."
    assert finals[0].author == "a"
    # a trivial (no-tool) turn yields exactly the one final agent event
    # (the user message is recorded in the session, not re-yielded)
    assert len(events) == 1


def test_function_tool_and_tool_context_state():
    def remember(fact: str, tool_context: ToolContext) -> dict:
        """Remember a fact in session state."""
        tool_context.state["fact"] = fact
        return {"status": "success"}

    agent = Agent(
        name="a", instruction="x", output_key="reply", tools=[remember],
        model=ScriptedBackend(
            'TOOL_CALL: {"tool": "remember", "args": {"fact": "sky is blue"}}',
            "noted"))
    runner = _make_runner(agent)
    list(runner.run(user_id="u", session_id="s", new_message="remember"))
    sess = runner.session_service.get_session(
        app_name="app", user_id="u", session_id="s")
    assert sess.state["fact"] == "sky is blue"   # tool wrote via ToolContext
    assert sess.state["reply"] == "noted"         # output_key persisted


def test_unknown_tool_is_handled_gracefully():
    agent = Agent(name="a", instruction="x", tools=[],
                  model=ScriptedBackend(
                      'TOOL_CALL: {"tool": "nope", "args": {}}', "fallback"))
    runner = _make_runner(agent)
    events = list(runner.run(user_id="u", session_id="s", new_message="go"))
    tool_evs = [e for e in events if e.event_kind() == "tool"]
    assert tool_evs and tool_evs[0].tool_result["status"] == "error"
    assert [e for e in events if e.is_final_response()][-1].text == "fallback"


# --- state prefix routing + data bus -------------------------------------

def test_state_prefix_routing():
    def setprefs(tool_context: ToolContext) -> dict:
        """Set a few state keys at different scopes."""
        tool_context.state["plain_key"] = "P"
        tool_context.state["user:pref"] = "dark"
        tool_context.state["temp:scratch"] = "ephemeral"
        return {"status": "success"}

    agent = Agent(name="a", instruction="x", tools=[setprefs],
                  model=ScriptedBackend(
                      'TOOL_CALL: {"tool": "setprefs", "args": {}}', "ok"))
    svc = InMemorySessionService()
    runner = Runner(agent=agent, app_name="app", session_service=svc)
    svc.create_session(app_name="app", user_id="u", session_id="s1")
    list(runner.run(user_id="u", session_id="s1", new_message="go"))

    s1 = svc.get_session(app_name="app", user_id="u", session_id="s1")
    assert "plain_key" in s1.state           # session-scoped → persisted
    assert "user:pref" not in s1.state       # routed to the user store
    assert "temp:scratch" not in s1.state    # temp: dropped, never persisted

    # user-scoped value is visible to a fresh session for the same user
    s2 = svc.create_session(app_name="app", user_id="u", session_id="s2")
    merged = svc.get_merged_state(s2)
    assert merged.get("user:pref") == "dark"
    assert "temp:scratch" not in merged
    assert "plain_key" not in merged         # session-scoped, not shared


def test_sequential_output_key_data_bus():
    a1 = Agent(name="a1", model=ScriptedBackend("ALPHA"),
               instruction="emit a token", output_key="a")
    # a2's model echoes whether a1's output reached its templated prompt.
    a2 = Agent(name="a2", instruction="Use {a} to answer.",
               model=(lambda prompt: "yes" if "ALPHA" in prompt else "no"))
    seq = SequentialAgent(name="seq", sub_agents=[a1, a2])
    runner = _make_runner(seq)
    finals = [e for e in runner.run(user_id="u", session_id="s",
                                    new_message="go") if e.is_final_response()]
    assert finals[-1].author == "a2"
    assert finals[-1].text == "yes"   # a2 saw a1's output via {a} state bus


def test_parallel_agent_branches():
    a1 = Agent(name="b1", model=ScriptedBackend("one"), instruction="x",
               output_key="o1")
    a2 = Agent(name="b2", model=ScriptedBackend("two"), instruction="x",
               output_key="o2")
    runner = _make_runner(ParallelAgent(name="par", sub_agents=[a1, a2]))
    events = list(runner.run(user_id="u", session_id="s", new_message="go"))
    sess = runner.session_service.get_session(
        app_name="app", user_id="u", session_id="s")
    assert sess.state["o1"] == "one" and sess.state["o2"] == "two"
    assert {e.branch for e in events if e.branch} == {"b1", "b2"}


# --- artifacts ------------------------------------------------------------

def test_artifacts_versioned_save_load_list():
    def save(name: str, body: str, tool_context: ToolContext) -> dict:
        """Save a small text artifact."""
        version = tool_context.save_artifact(
            name, body.encode("utf-8"), mime_type="text/plain")
        return {"status": "success", "version": version}

    agent = Agent(
        name="a", instruction="x", tools=[save], max_tool_iterations=4,
        model=ScriptedBackend(
            'TOOL_CALL: {"tool": "save", "args": {"name": "n.txt", "body": "v0"}}',
            'TOOL_CALL: {"tool": "save", "args": {"name": "n.txt", "body": "v1"}}',
            "saved"))
    runner = _make_runner(agent)
    list(runner.run(user_id="u", session_id="s", new_message="save twice"))
    svc = runner.artifact_service
    assert svc.list_artifact_keys(
        app_name="app", user_id="u", session_id="s") == ["n.txt"]
    v0 = svc.load_artifact(app_name="app", user_id="u", session_id="s",
                           filename="n.txt", version=0)
    latest = svc.load_artifact(app_name="app", user_id="u", session_id="s",
                               filename="n.txt")
    assert v0.data == b"v0"
    assert latest.data == b"v1" and latest.version == 1


# --- multi-agent composition ---------------------------------------------

def test_sub_agents_transfer():
    child = Agent(name="specialist", model=ScriptedBackend("Specialist answer."),
                  instruction="answer", description="handles specialist asks")
    parent = Agent(name="router", model=ScriptedBackend("TRANSFER: specialist"),
                   instruction="route", description="router", sub_agents=[child])
    runner = _make_runner(parent)
    events = list(runner.run(user_id="u", session_id="s",
                             new_message="need a specialist"))
    assert "transfer" in [e.event_kind() for e in events]
    final = [e for e in events if e.is_final_response()][-1]
    assert final.author == "specialist" and final.text == "Specialist answer."


def test_loop_agent_escalate_stops_early():
    runs = {"n": 0}

    def approve(tool_context: ToolContext) -> dict:
        """Approve and stop the loop."""
        runs["n"] += 1
        tool_context.actions.escalate = True
        return {"status": "success", "approved": True}

    critic = Agent(name="critic", instruction="approve", tools=[approve],
                   model=ScriptedBackend('TOOL_CALL: {"tool": "approve", "args": {}}'))
    runner = _make_runner(LoopAgent(name="refine", sub_agents=[critic],
                                    max_iterations=5))
    events = list(runner.run(user_id="u", session_id="s", new_message="go"))
    assert runs["n"] == 1                                   # stopped after 1 pass
    assert any(e.actions.escalate for e in events)


def test_loop_agent_runs_until_max():
    plain = Agent(name="echo", model=ScriptedBackend("tick"), instruction="x")
    runner = _make_runner(LoopAgent(name="loop", sub_agents=[plain],
                                    max_iterations=3))
    finals = [e for e in runner.run(user_id="u", session_id="s",
                                    new_message="go") if e.is_final_response()]
    assert len(finals) == 3


# --- callbacks ------------------------------------------------------------

def test_before_model_callback_short_circuits():
    class Boom:
        model, base_url = "boom", None

        def generate(self, *a, **k):
            raise AssertionError("model must not be called when short-circuited")

    agent = Agent(name="a", model=Boom(), instruction="x",
                  before_model_callback=lambda cb, prompt: "canned final")
    runner = _make_runner(agent)
    finals = [e for e in runner.run(user_id="u", session_id="s",
                                    new_message="hi") if e.is_final_response()]
    assert finals[-1].text == "canned final"


def test_before_tool_callback_short_circuits():
    called = {"n": 0}

    def real(x: str) -> dict:
        """A tool we expect to be bypassed."""
        called["n"] += 1
        return {"status": "success", "real": True}

    agent = Agent(
        name="a", instruction="x", tools=[real],
        before_tool_callback=lambda tool, args, tcx: {"status": "success",
                                                      "short": True},
        model=ScriptedBackend('TOOL_CALL: {"tool": "real", "args": {"x": "q"}}',
                              "done"))
    runner = _make_runner(agent)
    events = list(runner.run(user_id="u", session_id="s", new_message="go"))
    tool_evs = [e for e in events if e.event_kind() == "tool"]
    assert called["n"] == 0
    assert tool_evs[0].tool_result == {"status": "success", "short": True}


# --- the CoordPy guarantee underneath ------------------------------------

def test_capsule_trail_verifies_and_is_tamper_evident():
    agent = Agent(name="a", model=ScriptedBackend("answer"), instruction="x")
    runner = _make_runner(agent)
    list(runner.run(user_id="u", session_id="s", new_message="hi"))

    view = runner.session_capsule_view("s")
    assert view["schema"] == "coordpy.capsule_view.v1"
    assert view["chain_ok"] is True
    assert verify_chain_from_view_dict(view) is True
    assert runner.verify_session("s") is True
    assert len(view["capsules"]) >= 3   # user + final + run_report

    # corrupt a capsule CID -> chain recompute fails
    t1 = copy.deepcopy(view)
    t1["capsules"][1]["cid"] = "0" * 64
    assert verify_chain_from_view_dict(t1) is False
    # rewrite the chain head -> fails
    t2 = copy.deepcopy(view)
    t2["chain_head"] = "0" * 64
    assert verify_chain_from_view_dict(t2) is False


def test_deterministic_root_cid_for_identical_inputs():
    def build():
        def echo(x: str) -> dict:
            """Echo x."""
            return {"status": "success", "echo": x}

        agent = Agent(
            name="a", instruction="x", tools=[echo], output_key="ans",
            model=ScriptedBackend(
                'TOOL_CALL: {"tool": "echo", "args": {"x": "hi"}}', "done"))
        runner = InMemoryRunner(agent=agent, app_name="app")
        runner.session_service.create_session(
            app_name="app", user_id="u", session_id="sess")
        list(runner.run(user_id="u", session_id="sess", new_message="echo hi"))
        return runner.session_root_cid("sess")

    cid_a, cid_b = build(), build()
    assert cid_a and cid_a == cid_b   # content-addressed; no wall-clock in CID


def test_memory_add_and_search():
    agent = Agent(name="a", model=ScriptedBackend("Capsules are content-addressed."),
                  instruction="x")
    runner = _make_runner(agent)
    list(runner.run(user_id="u", session_id="s",
                    new_message="tell me about capsules"))
    sess = runner.session_service.get_session(
        app_name="app", user_id="u", session_id="s")
    runner.memory_service.add_session_to_memory(sess)
    resp = runner.memory_service.search_memory(
        app_name="app", user_id="u", query="capsules")
    assert len(resp.memories) >= 1
    assert any("capsule" in m.text.lower() for m in resp.memories)


# --- the bundled example app ---------------------------------------------

def test_research_assistant_example_runs():
    from coordpy.adk.examples.research_assistant import (
        build_research_assistant, root_agent, run_demo,
    )
    summary = run_demo()
    assert summary["capsule_view_verified"] is True
    assert summary["capsule_chain_ok"] is True
    assert summary["saved_artifacts"] == ["coordpy_in_brief.md"]
    assert "findings" in summary["state"] and "brief" in summary["state"]
    assert summary["memory_hits"] >= 1
    assert summary["capsules"] >= 5
    assert isinstance(root_agent, SequentialAgent)
    assert build_research_assistant().name == "research_assistant"
