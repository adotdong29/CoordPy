from __future__ import annotations

import dataclasses
import os
import unittest


@dataclasses.dataclass
class _FakeBackend:
    model: str = "fake.echo"
    base_url: str | None = None
    calls: list[str] = dataclasses.field(default_factory=list)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 80,
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(prompt)
        return f"out:{len(self.calls)}"


class CoordPyAgentsTests(unittest.TestCase):
    def test_create_team_runs_and_seals_capsules(self):
        from vision_mvp.coordpy import Agent, create_team

        backend = _FakeBackend()
        team = create_team(
            [
                Agent(name="planner", instructions="Plan the work."),
                Agent(name="writer", instructions="Write the answer."),
            ],
            backend=backend,
        )
        result = team.run("Summarize CoordPy.")
        self.assertEqual(result.final_output, "out:2")
        self.assertEqual(len(result.turns), 2)
        self.assertIsNotNone(result.capsule_view)
        self.assertTrue(result.capsule_view["chain_ok"])
        self.assertIsNotNone(result.root_cid)

    def test_visible_handoffs_are_bounded_to_avoid_token_cramming(self):
        from vision_mvp.coordpy import agent, create_team

        backend = _FakeBackend()
        team = create_team(
            [
                agent("a1", "First."),
                agent("a2", "Second."),
                agent("a3", "Third."),
            ],
            backend=backend,
            max_visible_handoffs=1,
            capture_capsules=False,
        )
        result = team.run("Do the task.")
        self.assertEqual(result.final_output, "out:3")
        third_prompt = backend.calls[-1]
        self.assertIn("bounded to avoid token cramming", third_prompt)
        self.assertIn("- a2: out:2", third_prompt)
        self.assertNotIn("- a1: out:1", third_prompt)

    def test_team_from_env_builds_openai_backend(self):
        from vision_mvp.coordpy import AgentTeam, agent, OpenAICompatibleBackend

        os.environ["COORDPY_BACKEND"] = "openai"
        os.environ["COORDPY_MODEL"] = "gpt-4o-mini"
        os.environ["COORDPY_API_KEY"] = "secret"
        try:
            team = AgentTeam.from_env(
                [agent("planner", "Plan the work.")],
                capture_capsules=False,
            )
            self.assertIsInstance(team.backend, OpenAICompatibleBackend)
            self.assertEqual(team.backend.model, "gpt-4o-mini")
        finally:
            del os.environ["COORDPY_BACKEND"]
            del os.environ["COORDPY_MODEL"]
            del os.environ["COORDPY_API_KEY"]

    def test_create_team_accepts_provider_kwargs(self):
        from vision_mvp.coordpy import create_team, agent, OpenAICompatibleBackend

        team = create_team(
            [agent("planner", "Plan the work.")],
            model="gpt-4o-mini",
            backend_name="openai",
            base_url="https://api.example.test",
            api_key="secret",
            capture_capsules=False,
        )
        self.assertIsInstance(team.backend, OpenAICompatibleBackend)
        self.assertEqual(team.backend.model, "gpt-4o-mini")
        self.assertEqual(team.backend.base_url, "https://api.example.test")

    def test_member_specific_backend_overrides_team_backend(self):
        from vision_mvp.coordpy import Agent, create_team

        team_backend = _FakeBackend(model="team")
        member_backend = _FakeBackend(model="member")
        team = create_team(
            [
                Agent(name="planner", instructions="Plan.", backend=member_backend),
                Agent(name="writer", instructions="Write."),
            ],
            backend=team_backend,
            capture_capsules=False,
        )
        result = team.run("Do it.")
        self.assertEqual(result.final_output, "out:1")
        self.assertEqual(len(member_backend.calls), 1)
        self.assertEqual(len(team_backend.calls), 1)


if __name__ == "__main__":
    unittest.main()
