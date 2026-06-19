"""Installed-wheel ADK product smoke test (W146).

This is the release gate's *product* check: it exercises the
``coordpy.adk`` front door against whatever ``coordpy`` is importable —
crucially the **installed wheel** when run from the release venv (it does
NOT inject the source tree onto ``sys.path``). The legacy
``tests/test_smoke_full.py`` covers the stable SDK contract; this file
covers the library-first ADK surface that real users import.

Run standalone (exit 0 on success, non-zero on first failure summary):

    python tests/test_adk_wheel_smoke.py

It is hermetic: a tiny scripted backend stands in for an LLM; no network,
no model, no API key.
"""

from __future__ import annotations

import importlib.metadata as _md

EXPECTED_VERSION = "1.2.1"

_failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK  " if cond else "FAIL"
    print(f"  [{status}] {name}{(' — ' + detail) if detail else ''}")
    if not cond:
        _failures.append(name)


class _ScriptedBackend:
    """Hermetic stand-in for an LLM: returns scripted text by call order."""

    def __init__(self, *script: str) -> None:
        self.model, self.base_url, self._i = "scripted", None, -1
        self._script = list(script)

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        self._i += 1
        return self._script[min(self._i, len(self._script) - 1)]


def main() -> int:
    print("# coordpy.adk installed-wheel smoke")

    # 1. Version parity: runtime __version__ ↔ installed distribution metadata.
    import coordpy
    check("import coordpy", True, coordpy.__file__)
    check(f"coordpy.__version__ == {EXPECTED_VERSION}",
          coordpy.__version__ == EXPECTED_VERSION, coordpy.__version__)
    try:
        dist_v = _md.version("coordpy-ai")
        check(f"dist metadata version == {EXPECTED_VERSION}", dist_v == EXPECTED_VERSION, dist_v)
    except _md.PackageNotFoundError:
        # Source/sys.path runs without an installed dist: skip parity, not a fail.
        print("  [SKIP] dist metadata version (coordpy-ai not installed as a dist)")

    # 2. The full ADK import surface resolves.
    import coordpy.adk as adk
    expected_adk = {
        "Agent", "LlmAgent", "BaseAgent",
        "SequentialAgent", "ParallelAgent", "LoopAgent",
        "BaseTool", "FunctionTool", "to_function_tool",
        "Runner", "InMemoryRunner", "ADK_RUN_REPORT_SCHEMA",
        "Session", "State", "Event", "EventActions",
        "BaseSessionService", "InMemorySessionService",
        "APP_PREFIX", "USER_PREFIX", "TEMP_PREFIX",
        "Artifact", "BaseArtifactService", "InMemoryArtifactService",
        "MemoryEntry", "SearchMemoryResponse",
        "BaseMemoryService", "InMemoryMemoryService",
        "ReadonlyContext", "CallbackContext", "ToolContext", "InvocationContext",
        "ADK_SURFACE_SCHEMA",
    }
    missing = sorted(n for n in expected_adk if not hasattr(adk, n))
    check("full coordpy.adk surface importable", not missing,
          f"missing={missing}" if missing else f"{len(expected_adk)} names")
    check("LlmAgent is Agent (alias)", adk.LlmAgent is adk.Agent)
    check("adk.Agent is distinct from legacy coordpy.Agent",
          adk.Agent is not coordpy.Agent)

    # 3. Schema constants.
    check("ADK_SURFACE_SCHEMA == coordpy.adk.v1",
          adk.ADK_SURFACE_SCHEMA == "coordpy.adk.v1", adk.ADK_SURFACE_SCHEMA)
    check("ADK_RUN_REPORT_SCHEMA == coordpy.adk.run.v1",
          adk.ADK_RUN_REPORT_SCHEMA == "coordpy.adk.run.v1",
          adk.ADK_RUN_REPORT_SCHEMA)

    # 4. Discoverability — the product front door must be in dir()/__all__.
    check("'adk' in dir(coordpy)", "adk" in dir(coordpy))
    check("'adk' in coordpy.__all__", "adk" in coordpy.__all__)
    check("__all__ is the curated stable surface (not the research dump)",
          len(coordpy.__all__) < 200, f"len={len(coordpy.__all__)}")

    # 5. Build → run → audit a real agent from the (installed) package.
    from coordpy.adk import Agent, InMemoryRunner, ToolContext

    def lookup_population(city: str, tool_context: ToolContext) -> dict:
        """Return the population of a city."""
        table = {"tokyo": 37_400_000}
        tool_context.state["last_city"] = city
        pop = table.get(city.lower())
        return ({"status": "success", "city": city, "population": pop}
                if pop else {"status": "error", "error": f"unknown {city!r}"})

    backend = _ScriptedBackend(
        'TOOL_CALL: {"tool": "lookup_population", "args": {"city": "Tokyo"}}',
        "Tokyo's metro population is about 37.4 million.",
    )
    assistant = Agent(name="geo_assistant", model=backend,
                      instruction="Answer population questions; use the tool.",
                      tools=[lookup_population], output_key="answer")
    runner = InMemoryRunner(agent=assistant, app_name="wheel_smoke")
    runner.session_service.create_session(
        app_name="wheel_smoke", user_id="u1", session_id="s1")

    final = None
    for event in runner.run(user_id="u1", session_id="s1",
                            new_message="How big is Tokyo?"):
        if event.is_final_response():
            final = event
    check("a final-response event was produced", final is not None)
    check("final answer is the scripted text",
          bool(final) and "37.4 million" in (final.text or ""),
          (final.text if final else ""))

    # 5b. The CoordPy guarantee underneath — re-verified from bytes.
    check("runner.verify_session('s1') is True",
          runner.verify_session("s1") is True)
    view = runner.session_capsule_view("s1")
    check("capsule view schema == coordpy.capsule_view.v1",
          view.get("schema") == "coordpy.capsule_view.v1", view.get("schema"))
    check("capsule chain_ok", bool(view.get("chain_ok")))
    check("legacy verifier accepts the ADK trail",
          coordpy.verify_chain_from_view_dict(view) is True)

    # 6. The documented packaged example runs end to end (ships in the wheel).
    from coordpy.adk.examples.research_assistant import run_demo
    s = run_demo()
    check("research_assistant: capsule_view_verified", s.get("capsule_view_verified") is True)
    check("research_assistant: capsule_chain_ok", s.get("capsule_chain_ok") is True)
    check("research_assistant: saved an artifact", len(s.get("saved_artifacts") or []) >= 1)
    check("research_assistant: memory hit", (s.get("memory_hits") or 0) >= 1)
    check("research_assistant: sealed >= 5 capsules", (s.get("capsules") or 0) >= 5)
    check("research_assistant: final answer non-empty", bool(s.get("final_answer")))

    print()
    if _failures:
        print(f"ADK WHEEL SMOKE FAILED: {len(_failures)} check(s): {_failures}")
        return 1
    print("ALL ADK WHEEL SMOKE CHECKS PASSED.")
    return 0


# pytest entry point (so `pytest tests/test_adk_wheel_smoke.py` works too).
def test_adk_wheel_smoke() -> None:
    assert main() == 0, f"ADK wheel smoke failures: {_failures}"


if __name__ == "__main__":
    raise SystemExit(main())
