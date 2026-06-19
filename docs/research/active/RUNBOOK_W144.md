# RUNBOOK W144 — CoordPy canonical-subject consolidation + main-tool tightening + executable subject harness

**Status: LOCKED before implementation.** Operator-greenlit bounded consolidation milestone.
Date 2026-06-12. Built on W143 (`428609e`, synced `466cf44`). `ultracode` OFF.
Stable boundary: `coordpy.__version__ == "0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`,
no PyPI publish, gate `258b6ed7` invariant.

> W144 is **NOT** a benchmark/margin/retirement milestone, **NOT** the separate architecture
> branch, and **NOT** a docs-only pass. W144 turns the repo into **one coherent, testable
> CoordPy subject**: decide the real front door, classify stable vs canonical-experimental vs
> historical, and land an executable subject harness in the main tool path.

## 0. One-line thesis

CoordPy already *is* a real product — a context-capsule runtime SDK/CLI — but its identity is
buried under ~871 `coordpy/*.py` modules, 378 docs, and a 700-line reverse-chronological
milestone dump. W144 makes "what is CoordPy, what is stable, what is canonical-experimental,
what is historical" answerable **in one executable run**, without archaeology.

## 1. Inputs that are already settled (measured this milestone, do not re-derive)

- **Canonical subject is already documented** (master plan §10.0; ARCHITECTURE.md): *"CoordPy is a
  context-capsule runtime"* — the **Capsule Contract C1–C6** is the SDK-v3 centre of gravity.
  Context Zero = the research programme; CoordPy = the shipped product. Do **not** flatten this split.
- **Front door is already documented**: `coordpy-team` is commented in `pyproject.toml` and in
  README/START_HERE as *"the recommended front door for new users."*
- **Census (this milestone, `/tmp/w144_census.json`, 871 modules)**: ~19–21 `__init__`-imported,
  ~755 live research leaves, ~93 script/doc-only, **only 4 orphans** (`__main__`, `_pretty` =
  legitimate internals; `hosted_cost_planner_v10`, `hosted_logprob_router_v10` = superseded
  versions in a v1..v12 ladder). **Conclusion: essentially nothing is dead to delete.**
- **Versioned ladders dominate**: `long_horizon_retention` (29), `persistent_latent` (28),
  `mergeable_latent_capsule` (27), `tiny_substrate`/`kv_bridge`/`deep_substrate_hybrid`/
  `consensus_fallback_controller` (24 each)… ≈400+ modules are W49–W66 substrate/dense-control
  version ladders → S4 historical.
- **Truth surface is clean** (docs scan): version (0.5.20/v3.43), front-door, retirement-count,
  and "current top marker = W143" are all consistent. The one real archaeology trap: ARCHITECTURE.md
  + ~68 historical docs reference stale `vision_mvp/...` paths (the package is top-level `coordpy/`).
- **graphify (HEAD `466cf445`, 87,284 nodes)**: `coordpy → multi_agent_discover_amortize_v1` =
  **no path** (W143 hub is a pure explicit-import leaf); `shared_state_proxy` is **1-hop from
  `agents.py`** (W48 north-star builds on the stable Agent surface); the W140–143 chain clusters in
  communities 4109/136/71, distinct from the stable core (4056/148/168) and the manifold lineage
  (12/21/143). Tiers are real graph structure.

## 2. Subject-classification rule (LOCKED before results)

Every examined `coordpy` surface lands in **exactly one** tier. The rule extends — does not replace —
the master plan §10.1 living stability matrix.

- **S1 stable-core** — on the released SDK/CLI public contract: imported by `coordpy/__init__.py`
  as a *stable* (non-`__experimental__`) symbol, **or** one of the 5 console scripts / 4 on-disk
  schemas / `_internal/product` modules / `extensions`. Test-locked by a `test_coordpy_*` contract test.
- **S2 canonical-experimental** — explicit-import-only, BUT central to CoordPy's *identity* by
  meeting ≥2 of: (a) named on the architecture north-star lineage (`Latent State Transition`:
  product_manifold/live_manifold/learned_manifold/autograd_manifold/shared_state_proxy); (b) named
  load-bearing in the **current** RESEARCH_STATUS top markers (W140–W143 discover-then-amortize
  chain); (c) has a dedicated test; (d) imported by another S2/recent module. Ships in the wheel,
  flagged unstable, reachable only via explicit import.
- **S3 benchmark/research-support** — the evaluation machinery: `*_bench_v*`, `*_corpus_v*`,
  `*_battlefield_v*`, `*_slate_v*`, `*_loader_v*`, `*_executor_v*`, `*_preflight_v*`, R-XX drivers,
  realworldqa/icpc/apps/bigcodebench/mbpp benches. Load-bearing to *results*, **not** to the product
  identity. Must stop presenting as central.
- **S4 historical/archive-only** — kept verbatim for provenance: the W22–W42 dense-control ladder
  (188 `__experimental__` symbols) and the W49–W139 substrate version ladders (`tiny_substrate_v*`,
  `persistent_latent_v*`, `kv_bridge_v*`, `long_horizon_retention_v*`, the `*_team`/`r*_benchmark`
  per-milestone modules). Explicit-import-only; never on the product front door.
- **S5 blocked/dead/keep-for-record** — explicitly killed or superseded: the W136-killed learned-memory
  trio (`differentiable_memory`, `composed_learned_memory`, `live_composed_*` where present),
  superseded version-ladder tails (`hosted_cost_planner_v10`, `hosted_logprob_router_v10`), and any
  module proven unreferenced. **Kept, not deleted** (provenance + no destructive churn).

Tie-break: a module that qualifies for two tiers takes the **most stable** tier it honestly meets,
except S1 which requires the `__init__` stable-symbol / console-script / schema test. The registry
records the rule output as data (`docs/W144_COORDPY_SUBJECT_REGISTRY.json`), falsifiable by re-running
the census + `coordpy-subject` harness.

### 2.1 Center-of-gravity framing brainstorm (done BEFORE locking the canonical subject)

**≥12 candidate framings:**
1. Context-capsule runtime (C1–C6 contract). ← anchor
2. Bounded-context-vs-token-cramming orchestration SDK.
3. Auditable/replayable multi-agent team runtime.
4. Content-addressed provenance ledger for LLM artefacts.
5. A capsule *schema* standard (on-disk `coordpy.*.v1` formats).
6. A multi-agent coordination research platform.
7. The "Latent State Transition" architecture-precursor lab.
8. A no-oracle self-tutoring / discover-then-amortize engine.
9. A contamination-resistant code-benchmark harness.
10. A trust-adjudication / dense-control ratification library (W22–W42).
11. A transformer-substrate proxy lab (W48–W66).
12. A profile-driven evaluation SDK (Slice-1 framing).
13. A CLI toolkit (`coordpy-*` scripts) for agent runs.
14. A reproducibility/CI gate for agent evals.

**≥8 ways a framing is too benchmark-shaped (kill for the *product* centre):** #8, #9 are milestone
results, not the product; #6 conflates programme with product; #12 is the superseded Slice-1
outcome-name (master plan §10.0 explicitly retired it); a framing centred on HumanEval/MBPP/ICPC
margins; on "retirements"; on NIM pilots; on pass@k. All are Context-Zero *programme* outputs — they
belong to S3, not the CoordPy identity.

**≥8 ways a framing is too architecture-shaped (kill for *this* milestone):** #7, #11 are the separate
architecture branch (explicitly out of scope); #10 (W22–W42 ratification) is S4 history; a framing on
KV-cache proxies / pseudo-attention / substrate coupling / learned manifolds as the *centre*; on
"Latent State Transition" as the product (it is the *north star of the next branch*, not the shipped
tool). All S2/S4, not S1.

**≥6 ways a framing is too diffuse to test:** #2 ("orchestration") and #3 ("auditable teams") are
outcome-shaped (master plan §10.0: not mechanism-shaped, can't be pinned); "an agent platform"
(explicitly disclaimed); "a context toolkit"; "an LLM framework"; "everything in the repo." None yield
a single executable contract.

**Survivor / canonical subject:** #1 **context-capsule runtime**, with #3/#4/#13/#14 as the
*operational surfaces* (team runtime, provenance/audit, CLI, CI gate) that the C1–C6 contract makes
coherent. This is mechanism-shaped, already documented, and directly testable by re-hashing a capsule
chain. **Locked.**

## 3. Front-door decision rule (LOCKED)

There are two honest, distinct questions; resolve the ambiguity by naming one canonical answer each,
and make the orientation door point at the usage door (so there is a single discovery path):

- **Canonical USAGE front door = `coordpy-team`** (unchanged; already documented). The first thing a
  user runs to *use* CoordPy.
- **Canonical ORIENTATION/SUBJECT front door = `coordpy-subject` (NEW, this milestone)**. The first
  thing a new agent runs to *understand and verify* CoordPy: it prints what stable CoordPy is, the
  S1–S5 tier map, what passes, what is out of scope, and explicitly names `coordpy-team` as the usage
  door. One run, deterministic report.

Rule: do not add a second *usage* front door; do not retire `coordpy-team`; the new surface is
strictly an orientation/harness tool that parallels the existing `coordpy-capsule` audit tool.
`coordpy` (profile runner), `coordpy-import`, `coordpy-ci` remain unchanged secondary tools.

## 4. Consolidation / harness slate (LOCKED before implementation)

1. **`coordpy/subject.py`** (NEW, S1-adjacent, additive). Pure-stdlib. Contains:
   - the **canonical subject statement** + the **S1–S5 registry** as in-code data;
   - `build_subject_report(run_checks: bool) -> dict` → deterministic `coordpy.subject.v1` report;
   - harness checks (all hermetic, no network): **stable SDK smoke** (import, version match,
     synthetic team run via `SyntheticLLMClient`), **capsule verification path**
     (`verify_chain_from_view_dict` round-trip), **team runtime path** (`create_team(...).run`
     sealing a capsule view), **curated S2 exemplar import-checks** (the 7 operator modules +
     manifold lineage importability), **schema-constant presence**;
   - `render_text(report)` for human output; JSON for machines.
2. **`coordpy/_cli.py`** (additive): `_cmd_subject` + `main_subject` thin wrapper. Subcommands:
   `coordpy-subject report` (default), `coordpy-subject check`, `coordpy-subject registry`,
   `coordpy-subject tiers`. `--json` / `--version` supported. Mirrors the existing `coordpy-capsule`
   subparser style. **Zero change to the 4 existing `main_*` entry points.**
3. **`pyproject.toml`** (additive): one new console script `coordpy-subject = "coordpy._cli:main_subject"`.
   `python -m coordpy.subject` also works (so it runs without reinstall; no republish needed).
4. **`docs/W144_COORDPY_SUBJECT_REGISTRY.json`** (NEW): the machine-readable S1–S5 registry +
   classification rule + census provenance. The harness reads/echoes its own embedded copy; the doc is
   the human-auditable mirror.
5. **`docs/RESULTS_W144_CANONICAL_SUBJECT_V1.md`** (NEW): lane-α audit writeup (framings, decision,
   tier map, evidence).
6. **`tests/test_w144_subject_harness_v1.py`** (NEW): locks the harness contract + the registry shape
   + that the 3 untested S2 chain modules (`self_tutoring_controller_v1`, `no_oracle_verifier_v2`,
   `multi_agent_discover_amortize_v1`) at least import + expose their documented symbols (closes a real
   coverage gap surfaced this milestone).
7. **Truth-surface tightening** (lane γ, see §8).

Hard limits: **no `coordpy/__init__.py` stable-surface export churn** beyond — at most — an *optional*
additive note; **no deletion** of historical surfaces; **no wrapping of 20 old scripts**; the harness
is a tool surface, not a milestone driver.

## 5. Validation rule (LOCKED)

W144 succeeds only if ALL hold:
1. `python -c "import coordpy; assert coordpy.__version__=='0.5.20' and coordpy.SDK_VERSION=='coordpy.sdk.v3.43'"` passes (boundary invariant).
2. `coordpy-subject report --json` and `python -m coordpy.subject` both run, exit 0, emit a
   `coordpy.subject.v1` JSON with all harness checks **PASS**.
3. `coordpy-subject check` returns non-zero **iff** a harness check fails (real gate, not cosmetic).
4. The new test file passes, AND a focused regression subset (capsule, public-API, team, agents,
   the W140–143 chain test) stays green.
5. The registry JSON parses, covers S1–S5, and its S1 list matches the `__init__` stable surface +
   console scripts (cross-checked against subagent-A extraction).
6. graphify rebuilt from the post-change HEAD; report shows the new commit.
7. No NIM / external model spend. No PyPI publish. No version bump.

## 6. Anti-overstatement rule (LOCKED)

- W144 is a **consolidation/legibility** milestone. It retires **nothing** and earns **no** new
  empirical claim. W89 (+5.56) + W105 (+7.00) remain the only two MULTI-AGENT retirements; W142b
  remains the distinct discover-then-amortize retirement; W143 closed the composition line.
- Do **not** claim the harness "proves CoordPy is production-ready," "solves multi-agent context," or
  "validates the research." It verifies the *stable contract smoke-tests pass and the subject is
  legible*, nothing more.
- The S2 "canonical-experimental" label is an *identity* judgement, not a stability promise; those
  modules stay explicit-import-only and unstable.
- Promotion of any module toward the main tool must be justified in §4/registry; absent justification,
  it stays explicit-import-only and the registry says so.
- Calling the repo "CoordPy" must not absorb the Context-Zero benchmark arc (S3) into the product.

## 7. graphify deliverables (LOCKED)

- Refresh `graphify update .` at **start** (done; built from `466cf445`) and at **end** (after code/doc
  changes); confirm `GRAPH_REPORT.md` "Built from commit" == post-change HEAD.
- Ran this milestone: `graphify explain` on coordpy/agents/capsule/capsule_runtime/team_coord/presets/
  llm_backend/shared_state_proxy/product_manifold/learned_manifold/self_tutoring_controller_v1/
  no_oracle_verifier_v2/family_tutor_compiler_v1/multi_agent_discover_amortize_v1; `graphify path`
  coordpy↔multi_agent_discover_amortize_v1 (no path), capsule_runtime↔self_tutoring_controller_v1
  (4 hops), agents↔shared_state_proxy (1 hop); `affected coordpy`.
- Findings feed the tier map (§2) and the registry (§4.4). `graphify query` only as a secondary
  claim-surface finder.

## 8. Truth-surface consolidation rule (LOCKED)

- Fix only the **canonical, load-bearing** docs; do **not** mass-rewrite the ~68 historical docs that
  carry stale `vision_mvp/...` paths (provenance + churn risk). Priority targets:
  - **ARCHITECTURE.md** — repoint the canonical capsule/test paths from `vision_mvp/coordpy/...` /
    `vision_mvp/tests/...` to the real top-level `coordpy/...` / `tests/...`; add a one-line
    "canonical subject + front doors" pointer to `coordpy-subject`.
  - **README.md / docs/START_HERE.md** — add a short "Orientation: run `coordpy-subject`" pointer and a
    crisp S1/S2/S3/S4 one-liner; keep the existing front-door + stable surface text.
  - **docs/RESEARCH_STATUS.md** — add a W144 consolidation marker (no new empirical claim).
  - **CHANGELOG.md** — one additive `coordpy-subject` entry under the unreleased/0.5.20 line; **no
    version bump**.
  - **docs/context_zero_master_plan.md §10.1** — add a row / note pointing to the W144 registry +
    `coordpy-subject` as the executable view of the stability matrix.
- The bar: a new agent can answer "what is CoordPy / what's the main tool / what's stable / experimental
  / historical" from START_HERE + one `coordpy-subject` run, without reading 700-line milestone dumps.
- Sync Linear: create the W144 issue under COO-6; update `linear_github_mapping.json`; post a closeout
  comment. `COO-9` stays the lead **research** path (W144 is orthogonal consolidation).

## 9. W145 branch logic (LOCKED)

At W144 close, the honest next move is one of (decided by operator):
- **(A) The separate architecture branch** — "Latent State Transition" — now that CoordPy is a legible,
  testable subject and the multi-agent composition line is closed (W143). This is the standing north
  star and the most likely W145.
- **(B) A code-competent local model** where the efficient form is NOT i.i.d.-reachable (re-opens
  discover-then-amortize under COO-9) — if an operator greenlights benchmark spend.
- **(C) Deepen the subject harness** — promote a curated S2 exemplar into a first-class
  `coordpy-subject demo`, or add capsule-replay differential checks — if consolidation value remains.
- **(D) Primary-KNOWN stronger model** on both battlefields if the `258b6ed7` gate opens.
W144 does **not** start the architecture branch; it makes the repo ready for it.
