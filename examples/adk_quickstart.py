"""coordpy.adk quickstart — the smallest useful agent, end to end.

Runs hermetically (a tiny scripted backend; no network, no model). To run it
against a real model, replace ``model=ScriptedBackend(...)`` with
``model=backend_from_env()`` and set the ``COORDPY_*`` environment variables
(see the README).

    python examples/adk_quickstart.py
"""

from __future__ import annotations

from coordpy.adk import Agent, InMemoryRunner, ToolContext


# 1. A tool is just a typed, docstring'd function. A trailing `tool_context`
#    parameter is auto-injected (state / artifacts / memory) and hidden from
#    the model-facing schema.
def lookup_population(city: str, tool_context: ToolContext) -> dict:
    """Return the population of a city."""
    table = {"tokyo": 37_400_000, "paris": 11_100_000, "lagos": 15_400_000}
    pop = table.get(city.lower())
    tool_context.state["last_city"] = city
    if pop is None:
        return {"status": "error", "error": f"unknown city {city!r}"}
    return {"status": "success", "city": city, "population": pop}


class ScriptedBackend:
    """A hermetic stand-in for an LLM (returns scripted text by call order)."""

    def __init__(self, *script: str) -> None:
        self.model, self.base_url, self._i = "scripted", None, -1
        self._script = list(script)

    def generate(self, prompt: str, max_tokens: int = 80,
                 temperature: float = 0.0) -> str:
        self._i += 1
        return self._script[min(self._i, len(self._script) - 1)]


def main() -> int:
    backend = ScriptedBackend(
        'TOOL_CALL: {"tool": "lookup_population", "args": {"city": "Tokyo"}}',
        "Tokyo's metro population is about 37.4 million.",
    )

    # 2. Define an agent: name + model + instruction + tools.
    assistant = Agent(
        name="geo_assistant",
        model=backend,
        instruction="Answer population questions; use the lookup tool.",
        tools=[lookup_population],
        output_key="answer",
    )

    # 3. Wire a Runner with an in-memory session service, and run one turn.
    runner = InMemoryRunner(agent=assistant, app_name="quickstart")
    runner.session_service.create_session(
        app_name="quickstart", user_id="u1", session_id="s1")

    # 4. A turn is a stream of events; the final one is is_final_response().
    for event in runner.run(user_id="u1", session_id="s1",
                            new_message="How big is Tokyo?"):
        if event.is_final_response():
            print("ANSWER:", event.text)

    # 5. The CoordPy guarantee underneath — for free:
    print("capsule chain verified from bytes:", runner.verify_session("s1"))
    view = runner.session_capsule_view("s1")
    print(f"sealed {len(view['capsules'])} content-addressed capsules "
          f"(schema {view['schema']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
