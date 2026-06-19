# RUNBOOK — W146: CoordPy 1.2.0 public-release hardening + repo reorganization + ADK-first docs + modern release gate

**Class:** public-release / packaging / repo-reorg milestone. **NOT** a benchmark milestone, **NOT** the architecture branch, **NOT** a docs-only pass, **NOT** a "same repo, higher version" release.
**Operator greenlight:** explicit (W146 prompt). `ultracode` OFF. Work on `main`.
**Baseline:** HEAD `093079d` (W145 landed on `origin/main`). Version `0.5.20` → **`1.2.0`**. SDK research tag `coordpy.sdk.v3.43` **unchanged**.
**Stable boundary held from W145:** CoordPy is a Python-first ADK; `coordpy.adk` is the product front door; capsule/provenance/replay is the guarantee underneath. W146 keeps that direction and does **not** touch the architecture branch (Latent State Transition) or the research frontier.

This RUNBOOK is **LOCKED before implementation**. It is the single decision path; deviations are recorded in `RESULTS_W146_*` and the CHANGELOG, not by silently editing this file.

---

## 1. α / β / γ branch logic

Three lanes, executed in a safety-ordered interleave (β's `src/` move first so every later edit lands on the final tree; α surface-shrink after the move; γ docs/gate last; verify gates throughout).

- **Lane α — public-release blocker cleanup + supported-surface tightening.** Scrub private infra from shipped package code; remove/relocate private ops files; shrink the root export surface; lock the 1.2.0 support boundary. **Succeeds iff** the package/repo become materially smaller + safer AND an explicit public support boundary is recorded.
- **Lane β — repo layout reorg + `src/` packaging.** `git mv coordpy src/coordpy`; adopt the taxonomy below; move clutter out of the root story; keep the product building + tests green. **Succeeds iff** the repo is materially cleaner, packaging builds from `src/`, and the product still builds/tests cleanly.
- **Lane γ — ADK-first docs/story + release-gate modernization.** Rewrite README/START_HERE/ARCHITECTURE/CHANGELOG ADK-first; normalize CONTRIBUTING/SECURITY/RELEASING/metadata to a semver 1.2.0 release; add an **installed-wheel ADK smoke** to the release gate; make `coordpy.adk` dominant in discovery. **Succeeds iff** real repo/package/test/release-gate changes land (not docs alone) and the new gates pass.

**Stop rule:** if any lane cannot complete cleanly, stop at the last green state and report the exact remaining blocker. Do **not** "fix after publish."

---

## 2. Public-release classification rule (locked BEFORE results)

Every examined surface lands in **exactly one** bucket:

1. **PUBLIC_PRODUCT** — shipped in the wheel + semver-supported: `coordpy.adk` (front door), the curated `coordpy` SDK (`_STABLE_PUBLIC`), the 6 console scripts, the 4 on-disk schemas, `py.typed`.
2. **PUBLIC_DOCS** — user-facing docs/examples: README, `docs/guides/*`, `docs/reference/*`, `docs/releases/*`, `examples/`.
3. **CONTRIBUTOR** — dev-only: CONTRIBUTING, `scripts/release/*`, `scripts/dev/*`, `tests/`, the release gate, `.github/workflows/{release,coordpy-ci}.yml`.
4. **RESEARCH_ARCHIVE** — preserved but out of the product path: `docs/archive/*`, `docs/research/*`, `papers/*`, the `coordpy.__experimental__` surface (ships for reproducibility/audit, importable by explicit path, **not** in `__all__`).
5. **PRIVATE_REMOVE** — out of the public tree entirely: LAN/aspen infra in package code, `.github/workflows/claude.yml`, Linear/GitHub PM sync (`linear_github_mapping.json`, `scripts/sync_linear_github_v1.py`, `docs/LINEAR_GITHUB_SYNC.md`), `TEST_CLAUDE.md`, tracked research artifacts (`results/`, `data/`), personal-path/LAN-IP docs/runbooks.

A file may be **relocated** (e.g. results → `artifacts/`) and then **gitignored**; "PRIVATE_REMOVE" means "stop tracking + leave the public tree," not necessarily "delete from disk."

---

## 3. Repo taxonomy + `src/` migration rule

Target tree:

```
/  .github/  docker/  src/coordpy/  tests/  examples/
   docs/{guides, reference, research/active, releases, archive}
   scripts/{release, dev, research}
   papers/{active, formal, archive}
   artifacts/{results, graphify, data, ops}   # gitignored
```

`src/` rule: a single `git mv coordpy src/coordpy`. Public import name stays `coordpy.*`. **No deep in-package renaming** beyond what `src/` requires. After the move, reinstall the editable install (`pip install -e ".[dev]"`) **before any test run** (the PEP 660 finder hardcodes the old path). Hazards fixed explicitly: `pyproject.toml` (`where=["src"]`, ruff/black/mypy path prefixes), `MANIFEST.in` (2 paths), `scripts/release/release.sh` (version grep path), `docker/Dockerfile.coordpy-substrate` (COPY path). The 27 `sys.path.insert(ROOT)` tests auto-resolve via the reinstall; the `[tool.setuptools.dynamic]` version attr is verified by a real `python -m build`.

`artifacts/` is **gitignored**; relocating tracked clutter there means `git rm --cached` + physical move + gitignore.

---

## 4. Public-surface support rule for 1.2.0

- **STABLE (semver-protected):** `coordpy.adk` (schema `coordpy.adk.v1`); the curated `coordpy` SDK surface (`_STABLE_PUBLIC`, now incl. `adk`); console scripts `coordpy`, `coordpy-team`, `coordpy-capsule`, `coordpy-subject`, `coordpy-import`, `coordpy-ci`; on-disk schemas `coordpy.capsule_view.v1`, `coordpy.team_result.v1`, `coordpy.provenance.v1`, `phase45.product_report.v2`. **Changing a schema or the capsule contract ⇒ major version bump** (carried from RELEASING.md).
- **EXPERIMENTAL (no stability promise):** `coordpy.__experimental__` + all research modules — ship in the wheel for reproducibility/audit, importable as `coordpy.<name>`, **removed from `__all__`** (so `from coordpy import *` and `dir(coordpy)` expose only the stable surface).
- **`__all__` shrinks** from 653 names to `list(_STABLE_PUBLIC)` (+`adk`). Safe: zero real `from coordpy import *` call sites in the repo; explicit `coordpy.<name>` imports of experimental names keep working.
- **`coordpy.adk` becomes discoverable**: added to `_STABLE_PUBLIC` so it appears in `dir(coordpy)` / IDE autocomplete and is the dominant front door.
- **`SDK_VERSION` (`coordpy.sdk.v3.43`)** stays, independent of the PyPI version (tracks the research line).
- **`coordpy-subject`** is retained as **contributor/orientation** diagnostics, not a primary user front door.

---

## 5. Release-gate modernization rule

- **Keep** the legacy `tests/test_smoke_full.py` (stable-SDK contract) in the gate for compatibility.
- **Add** an installed-wheel ADK smoke (`tests/test_adk_wheel_smoke.py`, standalone exit-code driver, imports the **installed** package — no `sys.path` injection) asserting: (a) `coordpy.__version__ == "1.2.0"` and `importlib.metadata.version("coordpy-ai") == "1.2.0"`; (b) the full `coordpy.adk` import set; (c) `ADK_SURFACE_SCHEMA == "coordpy.adk.v1"`; (d) build an `Agent` on a hermetic scripted backend, run via `InMemoryRunner`, assert a final-response event + `runner.verify_session(...) is True` + capsule chain ok; (e) `"adk" in dir(coordpy)` and `in coordpy.__all__`; (f) the packaged example `coordpy.adk.examples.research_assistant` runs end-to-end.
- **Wire** the ADK smoke + console-script checks (`coordpy --version`, `coordpy-team --help`, `coordpy-subject`) into `scripts/release/release.sh` `smoke()`, `.github/workflows/release.yml` (wheel step), and `.github/workflows/coordpy-ci.yml` (editable: run `pytest tests/test_w145_adk_surface_v1.py` + the packaged example).
- The release gate (`release.sh check`: build → twine check → check-wheel-contents → install wheel → legacy smoke → ADK wheel smoke) **must pass** before any tag/publish.

---

## 6. Anti-overstatement rule

- W146 is a **product/release-hardening** milestone. It earns **no** new empirical result and **retires nothing**.
- Frontier truth **unchanged**: **W89 (+5.56) + W105 (+7.00)** remain the only two multi-agent retirements; **W142b** the distinct discover-then-amortize retirement; the architecture branch is **untouched**; the stronger-model gate `258b6ed7` {KNOWN:1, UNKNOWN:4} is **untouched**.
- **`COO-9` remains the lead research path** (orthogonal to this product milestone).
- Do **not** claim "published to PyPI" unless `twine upload` actually ran and was verified from a fresh venv. If the gates pass but the upload is deferred, say **"tag-ready / release-candidate-ready, publication deferred"** explicitly.
- Do not claim the repo is "clean" beyond what landed; list what was removed/moved/kept.

---

## 7. Graphify deliverables

- **Start:** `graphify update .` ran at HEAD `093079d`; code graph re-extracted (AST 100% over 2499 files); "no topology changes" since the last full build (W145 tail commits were registry/docs, not code topology); the `adk` nodes are present. The cached `GRAPH_REPORT.md` freshness header still showed `aff4d702` because `update` leaves outputs untouched when topology is unchanged.
- **During:** used `graphify explain {coordpy, adk, subject}` + `graphify path` for the surface/discovery read.
- **End:** after the `src/` move + file relocations (which **do** change topology), run `graphify update .` again so `graphify-out/` matches repo truth; record the HEAD the end-graph is built from in `RESULTS_W146` and the final report. `graphify-out/` stays a local artifact (gitignored; not committed).

---

## 8. Truth-surface consolidation rule

Single sources of truth for W146:
- `docs/RUNBOOK_W146.md` (this file) — the locked contract → relocated to `docs/research/active/` with the latest runbooks.
- `docs/research/active/RESULTS_W146_*.md` — the result note (what landed, blockers found + handled, gate verdicts).
- `CHANGELOG.md` `## [1.2.0]` — the public, Keep-a-Changelog entry.
- Linear: a new W146 issue (child of `COO-6`, sibling of `COO-70`) + a closeout comment; closeout comments on `COO-6`/`COO-9` affirming COO-9 stays lead.
- The in-repo `linear_github_mapping.json` PM bridge is **removed from the public tree** (relocated to gitignored `artifacts/ops/`); the durable bridge becomes Linear + CHANGELOG + git history.

---

## 9. W147 branch logic

After W146 lands RC-ready (or published):
- **(a) Publish** 1.2.0 to PyPI if deferred (operator go) + post-release canary from a fresh venv; OR
- **(b) Deepen the ADK product** (persistent session/artifact/memory backends, streaming, an adk-native CLI); OR
- **(c) Begin the Latent State Transition architecture branch** (structural); OR
- **(d) Re-open `COO-9`** on greenlit benchmark spend (a code-competent model whose efficient form is not i.i.d.-reachable).

`COO-9` remains the lead **research** path regardless of which product/release branch W147 takes.

---

## Spend rules
No benchmark/NIM spend. No architecture-branch work. Local tests, wheel builds, install-from-wheel checks, release-gate runs allowed. **No PyPI publish and no `v1.2.0` tag push** in this milestone unless every gate passes, the repo is honestly public-ready, **and** the operator explicitly authorizes the irreversible upload; otherwise stop at tag-ready / RC-ready and say so.
