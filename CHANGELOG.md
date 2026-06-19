# Changelog

All notable changes to **coordpy-ai** are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project follows [semantic versioning](https://semver.org/spec/v2.0.0.html)
with one project rule: the report schema and the capsule contract are part of
the public surface, so a backwards-incompatible change to either requires a
major version bump (see [`RELEASING.md`](RELEASING.md)).

The research-line tag `coordpy.SDK_VERSION` (currently `coordpy.sdk.v3.43`)
tracks the underlying research programme and is **independent** of the PyPI
version. The complete milestone-by-milestone research history (W22..W145) is
preserved verbatim in
[`docs/research/active/milestone-log.md`](docs/research/active/milestone-log.md).

## [1.2.0] - 2026-06-19

First stable **public ADK release**. CoordPy is now a Python-first agent
development kit (`coordpy.adk`) with content-addressed capsule audit,
provenance, and replay underneath. This release hardens the repository and
package for open-source consumption. The capsule contract and the on-disk
schemas are unchanged.

The version jumps from `0.5.20` to `1.2.0` to mark the move from a research
preview to a stable, supported public library. `SDK_VERSION` stays
`coordpy.sdk.v3.43`.

### Added
- `coordpy.adk` is the primary, discoverable library front door: `Agent` /
  `LlmAgent`, `Runner` / `InMemoryRunner` (event-stream turn gated on
  `is_final_response()`), `Session` / `State`, `FunctionTool` / `ToolContext`,
  `SequentialAgent` / `ParallelAgent` / `LoopAgent`, `Event` / `EventActions`,
  and in-memory artifact + memory services. Surface schema
  `ADK_SURFACE_SCHEMA = "coordpy.adk.v1"`. `adk` is now in `dir(coordpy)` and
  `coordpy.__all__`. (The ADK surface first shipped during the 0.x line;
  1.2.0 promotes it to the supported front door.)
- Installed-wheel ADK smoke test (`tests/test_adk_wheel_smoke.py`) wired into
  the release gate (`scripts/release/release.sh`) and CI — the gate now
  validates the `coordpy.adk` surface from the *installed* wheel, runs the
  packaged `coordpy.adk.examples.research_assistant`, and checks the console
  scripts.

### Changed
- **`src/` layout.** The package now lives at `src/coordpy/`; the import name
  is unchanged (`coordpy`).
- **Repository reorganized** into a release-grade open-source layout:
  `docs/{guides,reference,research/active,releases,archive}`,
  `scripts/{release,dev,research}`, `papers/{active,formal,archive}`. Research
  outputs, the graph cache, and internal ops material now live under a
  gitignored `artifacts/` tree, out of the product path.
- **Smaller public surface.** `from coordpy import *` / `coordpy.__all__`
  shrank from 653 names to the curated ~93-name stable set. Research /
  experimental names remain importable as `coordpy.<name>` (and enumerated in
  `coordpy.__experimental__`) but are no longer in the wildcard or `dir()`
  surface and carry no stability promise.
- **ADK-first docs.** `README.md`, `docs/guides/start-here.md`, and
  `ARCHITECTURE.md` were rewritten as normal OSS library docs. The full
  research-programme architecture moved to
  `docs/research/active/ARCHITECTURE_VISION.md`; the milestone history moved
  to `docs/research/active/milestone-log.md`.
- `Development Status` classifier is now `5 - Production/Stable`.

### Removed
- **Private infrastructure scrubbed from shipped package code:** the
  `<lan-subnet>` MAC1/MAC2 endpoint special-casing in `coordpy.runtime` and
  the host-specific real-LLM profiles in `coordpy._internal.product.profiles` are gone,
  replaced by generic `local_*` profiles and the single `COORDPY_OLLAMA_URL`
  override. The published wheel no longer leaks any LAN topology.
- **Internal ops material removed from the public tree:** the Claude PR-bot
  GitHub workflow, the Linear↔GitHub PM mapping, the PM sync script, and the
  lab-ops runbooks were relocated to the gitignored `artifacts/ops/`.
- ~750 tracked research result files and stray scratch files were untracked
  from the public repository.

## [0.5.20] and earlier

The `0.5.x` line was the research-preview series (PyPI through `0.5.20`),
developed across the W22..W145 milestones. Those releases are summarized
verbatim, milestone by milestone, in
[`docs/research/active/milestone-log.md`](docs/research/active/milestone-log.md).
A representative earlier release note lives in
[`docs/releases/`](docs/releases/).
