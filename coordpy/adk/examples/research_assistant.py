"""A small multi-agent research / document-triage assistant.

A two-step team built with the ``coordpy.adk`` surface:

* **researcher** — searches a knowledge base (a ``search_docs`` tool) and
  writes its findings to session ``State`` via ``output_key="findings"``.
* **writer** — reads ``{findings}`` from state, composes a short brief, and
  saves it as a versioned **artifact** via a ``save_brief`` tool.

It is wired as a ``SequentialAgent`` (researcher → writer over shared state),
exercises a ``ToolContext`` (state writes + artifact saves), and — because it
runs on the ``coordpy.adk`` Runner — every step seals into a re-verifiable
capsule trail with no extra code.

Run it hermetically (no network, no model)::

    python -m coordpy.adk.examples.research_assistant

Run it against a real model::

    from coordpy.llm_backend import backend_from_env
    from coordpy.adk.examples.research_assistant import build_research_assistant
    root = build_research_assistant(model=backend_from_env())
"""

from __future__ import annotations

import json
from typing import Any

from coordpy.adk import (
    Agent, InMemoryRunner, SequentialAgent, ToolContext,
)

# A tiny in-repo "knowledge base" the researcher can search (no network).
_KNOWLEDGE_BASE: dict[str, str] = {
    "capsules": (
        "A CoordPy capsule is a typed, content-addressed, lifecycle-bounded "
        "unit of context. Its CID is the SHA-256 of (kind, payload, budget, "
        "parents), so any tampering breaks the hash chain."),
    "adk": (
        "coordpy.adk is a Python-first Agent Development Kit: build Agents, "
        "Tools, Sessions, and sub-agents and run them with a Runner; every "
        "step seals into an auditable capsule trail."),
    "provenance": (
        "Every CoordPy run produces a re-verifiable capsule view plus a "
        "manifest you can replay against another model."),
}


def search_docs(query: str) -> dict:
    """Search the knowledge base for snippets relevant to `query`."""
    q = {w for w in query.lower().replace("?", " ").split()}
    matches = []
    for key, text in _KNOWLEDGE_BASE.items():
        overlap = q & {w for w in (key + " " + text).lower().split()}
        if overlap:
            matches.append({"topic": key, "snippet": text})
    return {"status": "success", "matches": matches or [
        {"topic": "none", "snippet": "no matching documents"}]}


def save_brief(title: str, body: str, tool_context: ToolContext) -> dict:
    """Save a brief as a versioned markdown artifact and record its name."""
    filename = (
        "".join(c if c.isalnum() else "_" for c in title.lower())[:40]
        + ".md")
    content = f"# {title}\n\n{body}\n".encode("utf-8")
    version = tool_context.save_artifact(filename, content, mime_type="text/markdown")
    tool_context.state["brief_artifact"] = filename
    return {"status": "success", "filename": filename, "version": version}


def build_research_assistant(model: Any = None) -> SequentialAgent:
    """Build the researcher→writer assistant. ``model`` is any coordpy
    ``LLMBackend`` / callable / model-tag; ``None`` resolves from the
    environment at run time."""
    researcher = Agent(
        name="researcher",
        model=model,
        instruction="Find the facts that answer the user's question. Use the "
                    "search_docs tool, then report the key findings plainly.",
        tools=[search_docs],
        output_key="findings",
        description="Searches the knowledge base and reports findings.",
    )
    writer = Agent(
        name="writer",
        model=model,
        instruction="Write a concise brief answering the user, grounded in "
                    "these findings:\n{findings}\nThen save it with save_brief.",
        tools=[save_brief],
        output_key="brief",
        description="Writes and files the final brief.",
    )
    return SequentialAgent(
        name="research_assistant",
        description="A two-step research + document-triage assistant.",
        sub_agents=[researcher, writer])


class _DemoBackend:
    """A deterministic, hermetic backend that drives the demo flow.

    Returns scripted responses by call order (researcher: search then report;
    writer: save then confirm), falling back to a plain answer if overrun.
    """

    def __init__(self) -> None:
        self.model = "coordpy.adk.demo"
        self.base_url = None
        self._i = -1
        self._script = [
            'TOOL_CALL: {"tool": "search_docs", "args": {"query": "capsules adk"}}',
            "Findings: CoordPy models context as content-addressed capsules; "
            "coordpy.adk is its Python-first ADK front door.",
            'TOOL_CALL: {"tool": "save_brief", "args": {"title": "CoordPy in brief", '
            '"body": "CoordPy is a context-capsule runtime with a Python-first ADK."}}',
            "Brief saved. CoordPy is a context-capsule runtime; its capsules make "
            "every agent handoff auditable, and coordpy.adk is the library front door.",
        ]

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        self._i += 1
        if self._i < len(self._script):
            return self._script[self._i]
        return "Done."


def run_demo(model: Any = None) -> dict:
    """Run the assistant once and return a structured summary of what
    happened (final answer, saved artifacts, and the capsule audit)."""
    assistant = build_research_assistant(model=model or _DemoBackend())
    runner = InMemoryRunner(agent=assistant, app_name="research_assistant")
    runner.session_service.create_session(
        app_name="research_assistant", user_id="demo_user", session_id="demo")

    final_text = ""
    kinds: list[str] = []
    for event in runner.run(user_id="demo_user", session_id="demo",
                            new_message="What is CoordPy and why capsules?"):
        kinds.append(event.event_kind())
        if event.is_final_response():
            final_text = event.text

    session = runner.session_service.get_session(
        app_name="research_assistant", user_id="demo_user", session_id="demo")
    artifacts = runner.artifact_service.list_artifact_keys(
        app_name="research_assistant", user_id="demo_user", session_id="demo")

    # Long-term memory: ingest this session, then recall it.
    runner.memory_service.add_session_to_memory(session)
    recall = runner.memory_service.search_memory(
        app_name="research_assistant", user_id="demo_user", query="capsules")

    view = runner.session_capsule_view("demo")
    return {
        "final_answer": final_text,
        "event_kinds": kinds,
        "saved_artifacts": artifacts,
        "state": dict(session.state),
        "memory_hits": len(recall.memories),
        "capsules": len(view["capsules"]),
        "capsule_chain_ok": view["chain_ok"],
        "capsule_view_verified": runner.verify_session("demo"),
        "root_cid": runner.session_root_cid("demo"),
    }


# Module-level root agent for ADK-style discovery (resolves model from env).
root_agent = build_research_assistant()


def main() -> int:
    summary = run_demo()
    print("=== coordpy.adk research assistant (hermetic demo) ===\n")
    print("Final answer:\n  " + summary["final_answer"] + "\n")
    print("Event stream: " + " -> ".join(summary["event_kinds"]))
    print("Saved artifacts: " + ", ".join(summary["saved_artifacts"]))
    print("Session state keys: " + ", ".join(sorted(summary["state"])))
    print(f"Long-term memory hits for 'capsules': {summary['memory_hits']}")
    print(
        f"\nCapsule audit (free, underneath): {summary['capsules']} capsules; "
        f"chain_ok={summary['capsule_chain_ok']}; "
        f"verified_from_bytes={summary['capsule_view_verified']}")
    print("root_cid=" + str(summary["root_cid"]))
    print("\nFull summary JSON:\n" + json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
