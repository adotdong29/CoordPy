"""Layered CoordPy API — three tiers for three audiences.

A single runtime can serve very different users:

  * **End users** want to run a named profile and get a report.
  * **Developers** want a programmatic builder that lets them assemble
    a run from parts without learning the underlying substrate.
  * **Researchers** want raw access to the capsule primitives
    (``CapsuleLedger``, ``ContextCapsule``) so they can wire in new
    admission policies, decoders, or audit passes.

This module exposes three thin classes — ``CoordPySimpleAPI``,
``CoordPyBuilderAPI``, ``CoordPyAdvancedAPI`` — plus a tiny stable
team/agent-spec layer that routes to the same underlying primitives
but presents them at different cognitive cost. No new runtime logic
lives here; these are just ergonomic surfaces over modules that
already exist.

See ``ADVANCEMENT_TO_10_10.md`` Part III §4 for the design note.
"""
from __future__ import annotations

import dataclasses
from typing import Any

from .capsule import (
    CapsuleBudget,
    CapsuleKind,
    CapsuleLedger,
    ContextCapsule,
    render_view,
)
from .run import RunSpec, run


@dataclasses.dataclass(frozen=True)
class AgentRoleSpec:
    """Human-readable definition of one agent in a CoordPy team."""

    name: str
    role: str | None = None
    instructions: str = ""
    budget: int = 2048
    model: str | None = None


@dataclasses.dataclass(frozen=True)
class AgentTeamSpec:
    """Small stable team definition for developer-facing code.

    This is intentionally additive. It gives downstream code a clear,
    typed way to talk about "my team" and "my agent roles" without
    changing the existing run-boundary contract. ``to_run_spec()``
    still returns an ordinary ``RunSpec``.
    """

    name: str
    mission: str = ""
    roles: tuple[AgentRoleSpec, ...] = ()

    def to_run_spec(
            self,
            *,
            profile: str,
            out_dir: str,
            acknowledge_heavy: bool = False,
            config: "Any | None" = None,
            ) -> RunSpec:
        return RunSpec(
            profile=profile,
            out_dir=out_dir,
            acknowledge_heavy=acknowledge_heavy,
            config=config,
        )


class AgentTeamBuilder:
    """Small fluent builder for "create agents / teams" use cases."""

    def __init__(self, name: str, *, mission: str = "") -> None:
        self._name = name
        self._mission = mission
        self._roles: list[AgentRoleSpec] = []

    def add_agent(
            self,
            name: str,
            *,
            role: str | None = None,
            instructions: str = "",
            budget: int = 2048,
            model: str | None = None,
            ) -> "AgentTeamBuilder":
        if budget <= 0:
            raise ValueError(f"agent budget must be > 0, got {budget}")
        self._roles.append(AgentRoleSpec(
            name=name,
            role=role,
            instructions=instructions,
            budget=int(budget),
            model=model,
        ))
        return self

    def build(self) -> AgentTeamSpec:
        return AgentTeamSpec(
            name=self._name,
            mission=self._mission,
            roles=tuple(self._roles),
        )


# ---------------------------------------------------------------------------
# Tier 1 — CoordPySimpleAPI: "one-call" for end users.
# ---------------------------------------------------------------------------


class CoordPySimpleAPI:
    """High-level, one-call CoordPy surface.

    **Audience**: end users who want to execute a named profile and
    get a product report back. No capsule, budget, or ledger concepts
    are exposed. Every field has a sensible default.

    Example::

        api = CoordPySimpleAPI()
        report = api.run_profile("local_smoke", out_dir="/tmp/coordpy-out")
        assert report["readiness"]["ready"]
    """

    def run_profile(self,
                    profile: str,
                    out_dir: str,
                    *,
                    acknowledge_heavy: bool = False) -> dict[str, Any]:
        """Run a registered profile; return the product report dict."""
        spec = RunSpec(profile=profile, out_dir=out_dir,
                       acknowledge_heavy=acknowledge_heavy)
        return run(spec)

    def team(self, name: str, *, mission: str = "") -> AgentTeamBuilder:
        """Create a small stable team-builder for developer code."""
        return AgentTeamBuilder(name, mission=mission)

    def run_team(self,
                 team: AgentTeamSpec,
                 *,
                 profile: str,
                 out_dir: str,
                 acknowledge_heavy: bool = False,
                 config: "Any | None" = None) -> dict[str, Any]:
        """Execute a team through the same stable run boundary."""
        spec = team.to_run_spec(
            profile=profile,
            out_dir=out_dir,
            acknowledge_heavy=acknowledge_heavy,
            config=config,
        )
        return run(spec)


# ---------------------------------------------------------------------------
# Tier 2 — CoordPyBuilderAPI: fluent configuration for developers.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BuilderSpec:
    """Resolved output of a ``CoordPyBuilderAPI.build()`` call.

    A plain dataclass so callers can inspect it, serialise it, or
    forward it into whatever runner they already have. ``to_run_spec``
    produces a ``RunSpec`` suitable for ``coordpy.run(...)``.
    """

    profile: str
    out_dir: str
    roles: tuple[tuple[str, int], ...] = ()  # (role_name, token_budget)
    team: AgentTeamSpec | None = None
    compression: str = "medium"  # "low" | "medium" | "high"
    acknowledge_heavy: bool = False
    model: str | None = None
    backend: str | None = None
    base_url: str | None = None

    def to_run_spec(self) -> RunSpec:
        config = None
        if self.model or self.backend or self.base_url:
            from .config import CoordPyConfig
            config = CoordPyConfig(
                model=self.model,
                llm_backend=self.backend,
                llm_base_url=self.base_url,
            )
        return RunSpec(
            profile=self.profile, out_dir=self.out_dir,
            acknowledge_heavy=self.acknowledge_heavy,
            config=config)


class CoordPyBuilderAPI:
    """Fluent builder for developers assembling a run from parts.

    **Audience**: application developers who want a programmatic
    configuration surface but do not need to touch capsule primitives.
    The builder stores partial state and produces a ``BuilderSpec``
    via ``.build()``. Method calls are chainable.

    Example::

        spec = (CoordPyBuilderAPI()
                .with_profile("local_smoke")
                .with_out_dir("/tmp/coordpy-out")
                .with_role("code_writer", budget=4096)
                .with_compression("medium")
                .build())
        report = spec.to_run_spec()  # feed into coordpy.run
    """

    _ALLOWED_COMPRESSION = frozenset({"low", "medium", "high"})

    def __init__(self) -> None:
        self._profile: str | None = None
        self._out_dir: str | None = None
        self._roles: list[tuple[str, int]] = []
        self._team_name: str | None = None
        self._team_mission: str = ""
        self._agents: list[AgentRoleSpec] = []
        self._compression: str = "medium"
        self._ack_heavy: bool = False
        self._model: str | None = None
        self._backend: str | None = None
        self._base_url: str | None = None

    def with_profile(self, profile: str) -> "CoordPyBuilderAPI":
        self._profile = profile
        return self

    def with_out_dir(self, out_dir: str) -> "CoordPyBuilderAPI":
        self._out_dir = out_dir
        return self

    def with_role(self, role: str, *, budget: int) -> "CoordPyBuilderAPI":
        if budget <= 0:
            raise ValueError(f"role budget must be > 0, got {budget}")
        self._roles.append((role, int(budget)))
        return self

    def with_team(self, name: str, *, mission: str = "") -> "CoordPyBuilderAPI":
        self._team_name = name
        self._team_mission = mission
        return self

    def with_agent(self,
                   name: str,
                   *,
                   role: str | None = None,
                   instructions: str = "",
                   budget: int = 2048,
                   model: str | None = None) -> "CoordPyBuilderAPI":
        if budget <= 0:
            raise ValueError(f"agent budget must be > 0, got {budget}")
        self._agents.append(AgentRoleSpec(
            name=name,
            role=role,
            instructions=instructions,
            budget=int(budget),
            model=model,
        ))
        return self

    def with_compression(self, level: str) -> "CoordPyBuilderAPI":
        if level not in self._ALLOWED_COMPRESSION:
            raise ValueError(
                f"compression must be one of {sorted(self._ALLOWED_COMPRESSION)}")
        self._compression = level
        return self

    def with_heavy_acknowledged(self, ack: bool = True) -> "CoordPyBuilderAPI":
        self._ack_heavy = bool(ack)
        return self

    def with_model(self, model: str) -> "CoordPyBuilderAPI":
        self._model = model
        return self

    def with_backend(self,
                     backend: str,
                     *,
                     base_url: str | None = None) -> "CoordPyBuilderAPI":
        self._backend = backend
        if base_url is not None:
            self._base_url = base_url
        return self

    def build(self) -> BuilderSpec:
        if self._profile is None:
            raise ValueError("profile is required; call with_profile(...)")
        if self._out_dir is None:
            raise ValueError("out_dir is required; call with_out_dir(...)")
        team = None
        if self._team_name is not None:
            team = AgentTeamSpec(
                name=self._team_name,
                mission=self._team_mission,
                roles=tuple(self._agents),
            )
        return BuilderSpec(
            profile=self._profile,
            out_dir=self._out_dir,
            roles=tuple(self._roles),
            team=team,
            compression=self._compression,
            acknowledge_heavy=self._ack_heavy,
            model=self._model,
            backend=self._backend,
            base_url=self._base_url,
        )


# ---------------------------------------------------------------------------
# Tier 3 — CoordPyAdvancedAPI: direct access to capsule substrate.
# ---------------------------------------------------------------------------


class CoordPyAdvancedAPI:
    """Low-level capsule primitives for researchers.

    **Audience**: researchers and systems engineers who want to
    construct capsules, admit them into a ledger, and render views
    directly. This surface is ``CapsuleLedger`` and ``ContextCapsule``
    with a convenience wrapper; it is NOT new runtime logic.

    Example::

        api = CoordPyAdvancedAPI()
        cap = api.make_capsule(
            kind="HANDOFF",
            payload={"msg": "hello"},
            budget=CapsuleBudget(max_tokens=128, max_bytes=1024))
        sealed = api.admit_and_seal(cap)
        assert api.ledger.verify_chain()
    """

    def __init__(self, ledger: CapsuleLedger | None = None) -> None:
        self.ledger: CapsuleLedger = ledger or CapsuleLedger()

    def make_capsule(self, *, kind: str, payload: Any,
                      budget: CapsuleBudget,
                      parents: tuple[str, ...] = ()) -> ContextCapsule:
        if kind not in CapsuleKind.ALL:
            raise ValueError(
                f"unknown kind {kind!r}; must be one of {sorted(CapsuleKind.ALL)}")
        return ContextCapsule.new(
            kind=kind, payload=payload, budget=budget, parents=parents)

    def admit_and_seal(self, capsule: ContextCapsule) -> ContextCapsule:
        return self.ledger.admit_and_seal(capsule)

    def view(self, *, include_payload: bool = False,
              root_cid: str | None = None) -> dict[str, Any]:
        return render_view(
            self.ledger, include_payload=include_payload,
            root_cid=root_cid).as_dict()


__all__ = [
    "AgentRoleSpec",
    "AgentTeamSpec",
    "AgentTeamBuilder",
    "CoordPySimpleAPI",
    "CoordPyBuilderAPI",
    "CoordPyAdvancedAPI",
    "BuilderSpec",
]
