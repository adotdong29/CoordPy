# RESULTS — W146: CoordPy 1.2.0 public-release hardening + repo reorg + ADK-first docs + modern release gate

**Class:** public-release / packaging / repo-reorganization milestone. Earns **no** new empirical result; **retires nothing**. No benchmark/NIM spend. Executes the locked `docs/research/active/RUNBOOK_W146.md` (α/β/γ). `ultracode` OFF. Work on `main`.

**One line.** CoordPy is now packaged and laid out as a real public 1.2.0 ADK library: private LAN/ops infra removed from the shipped path, a `src/` layout, a curated ~93-name public surface around `coordpy.adk`, ADK-first docs, and a release gate that validates the installed wheel — all green; publication is **earned but intentionally deferred** to an explicit operator go.

## Frontier truth (unchanged)
W89 (+5.56) + W105 (+7.00) remain the only two multi-agent retirements; W142b the distinct discover-then-amortize retirement. The Latent State Transition architecture branch was **not** started. The stronger-model gate `258b6ed7` {KNOWN:1, UNKNOWN:4} is untouched. **`COO-9` remains the lead research path** (W146 is an orthogonal product/release milestone).

## Lane α — blocker cleanup + supported-surface tightening
- **Private infra scrubbed from shipped package code** (the only leak reaching PyPI artifacts): `coordpy.runtime._resolve_endpoint` lost the `192.168.12.191/.248` MAC1/MAC2 special-casing (now the single `COORDPY_OLLAMA_URL` override); `coordpy._internal.product.profiles` `aspen_mac1_coder` / `aspen_mac2_frontier` / `aspen_mac1_coder_70b` → generic `local_coder` / `local_frontier` / `local_coder_70b` on `http://localhost:11434`; one `team_coord` docstring IP genericized. **Verified: the 1.2.0 wheel and sdist contain zero `192.168.*` / `aspen` / MAC infra.** (The `differential_privacy_v1` `192.168.1.42` is a PII-redaction demo fixture — correctly kept.)
- **Export surface shrunk** from **653 → 93** names: `coordpy.__all__ = list(_STABLE_PUBLIC)`. `adk` added to `_STABLE_PUBLIC`, so the product front door is now in `dir(coordpy)` and `__all__`. Experimental names remain importable as `coordpy.<name>` (and enumerated in `coordpy.__experimental__`) but are out of the wildcard / `dir()` surface. Safe: zero real `from coordpy import *` call sites.
- **Private ops removed from the public tree:** `.github/workflows/claude.yml` (PR-bot) and `TEST_CLAUDE.md` deleted; the Linear↔GitHub mapping, the PM sync script, and the lab-ops runbooks (`LINEAR_GITHUB_SYNC.md`, `MLX_DISTRIBUTED_RUNBOOK.md`, `COLAB_PRO_RUNBOOK.md`) relocated to the gitignored `artifacts/ops/`.
- **1.2.0 support boundary (locked):** STABLE = `coordpy.adk` (`coordpy.adk.v1`) + the curated `coordpy` SDK + the 6 console scripts + the 4 on-disk schemas (schema/capsule-contract change ⇒ major bump). EXPERIMENTAL = `coordpy.__experimental__` + the research ladder (ships for reproducibility, no stability promise). `SDK_VERSION = coordpy.sdk.v3.43` stays, independent of the PyPI version.

## Lane β — repo layout reorg + `src/` packaging
- `git mv coordpy src/coordpy` (945 files). Import name unchanged. `pyproject.toml` `where=["src"]` + ruff/black/mypy path prefixes + `pytest pythonpath=["src"]`; `MANIFEST.in`, `scripts/release/release.sh` (incl. `ROOT=../..` two-up fix), and `docker/Dockerfile.coordpy-substrate` updated. The `[tool.setuptools.dynamic]` version attr resolves under `src/` (verified by a real `python -m build` → `1.2.0`).
- **Taxonomy landed:** `docs/{guides,reference,research/active,releases,archive}` (≈340 milestone artifacts → `archive/milestones`, 119 phase JSONs → `archive/data`); `scripts/{release(2),dev(5),research(190)}`; `papers/{active,formal,archive}`; research outputs / graph cache / ops → gitignored `artifacts/{results,data,graphify,ops}`.
- **Untracked clutter:** ~749 `results/` + 3 `data/` files untracked & relocated; root junk (`out.log`, `data.json`, `df_contents.txt`, `mock_output.json`) and stray test dirs removed; `.gitignore` extended (`/artifacts/`, `graphify-out/`, `.hypothesis/`, …). Root now contains only standard OSS entries + `src/`.
- **Version bump 0.5.20 → 1.2.0:** `src/coordpy/_version.py`, `coordpy.subject.EXPECTED_VERSION`, and ~21 research-milestone "no-version-bump-invariant" self-checks/test assertions retargeted to `1.2.0` (historical "the 0.5.20 wheel froze X" docstrings left intact).

## Lane γ — ADK-first docs + release-gate modernization
- **Docs rewritten ADK-first:** `README.md` (version + links), `docs/guides/start-here.md` (934-line milestone dump → ~130-line onboarding), `ARCHITECTURE.md` (1477-line research/phase doc → 245-line product architecture; the full version preserved at `docs/research/active/ARCHITECTURE_VISION.md`), `CHANGELOG.md` (38 research blobs → Keep-a-Changelog `[1.2.0]`; full prose preserved at `docs/research/active/milestone-log.md`), plus `SECURITY.md` (1.2.x + all CLIs), `CONTRIBUTING.md` (ADK-first surface + `src/`/script paths), `RELEASING.md` (paths). `Development Status` → Production/Stable.
- **Release gate modernized:** new `tests/test_adk_wheel_smoke.py` — an installed-wheel ADK smoke (version+metadata parity, the full 33-name `coordpy.adk` surface, `ADK_SURFACE_SCHEMA`/`ADK_RUN_REPORT_SCHEMA`, `adk` in `dir()`/`__all__`, build→run→`verify_session` a hermetic agent, the packaged `research_assistant` example). Wired into `scripts/release/release.sh smoke()`, `.github/workflows/release.yml` (wheel step), and `coordpy-ci.yml` (pytest the ADK surface + run the example + `coordpy-subject check`).

## Gate verdicts
- `scripts/release/release.sh check` → **ALL GREEN** (build sdist+wheel `1.2.0`, `twine check`, `check-wheel-contents`, install wheel into a clean venv, legacy `test_smoke_full.py`, the new ADK wheel smoke (23/23), `coordpy --version`, `coordpy-team --help`, `coordpy --profile local_smoke`, `coordpy-capsule verify`).
- Focused product tests pass: `test_smoke_full`, `test_w144_subject_harness_v1` (registry path retargeted to `docs/reference/`), `test_w145_adk_surface_v1`, `test_adk_wheel_smoke`.
- **Wheel/sdist verified leak-free** of LAN/aspen/results/ops material; wheel ships `coordpy.adk` (11 modules) + the packaged example + `py.typed` + task data.

## Known, out-of-scope debt (transparent)
- `ruff check .` reports ~2.1k errors, almost all in the **research-ladder modules** (`src/coordpy/*_v1.py`, `r*_benchmark.py`, manifold/substrate) and research tests/scripts. This is **pre-existing** (W146 only relocated those files) and **not an enforced gate** (neither CI, `release.yml`, nor `release.sh` run ruff). A research-code lint cleanup is a separate effort, deliberately not bundled into this release-hardening milestone.
- Some research-archive docs still contain intra-doc links to the old `docs/START_HERE.md` path; the live user path (README → `docs/guides/start-here.md`) is correct.

## Disposition
The repo is **honestly ready for a public 1.2.0 release candidate**: it builds, installs from the wheel, passes the modernized gate, and the published artifacts are private-infra-free. **PyPI publication is NOT performed in W146** — pushing a `v1.2.0` tag auto-triggers the Trusted-Publisher upload, which is irreversible and outward-facing, so it is left for an explicit operator go (`RELEASING.md`). **No `v1.2.0` tag was pushed.**

## W147
(a) Publish 1.2.0 to PyPI (operator go) + post-release fresh-venv canary; OR (b) deepen the ADK product (persistent backends / streaming / adk-native CLI); OR (c) begin the Latent State Transition architecture branch; OR (d) re-open `COO-9` on greenlit benchmark spend. `COO-9` stays the lead research path.
