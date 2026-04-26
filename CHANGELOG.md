# Changelog

The Changelog now tracks **Wevra SDK** releases. The research
programme's phase-by-phase narrative lives in
`vision_mvp/RESULTS_PHASE*.md` and
`docs/context_zero_master_plan.md`.

## [3.10] — 2026-04-26 — SDK v3.10 — multi-service top-K cross-role corroboration multi-agent coordination + W9 family

*Strictly additive on SDK v3.9. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``MultiServiceCorroborationAdmissionPolicy`` is a research-slice
addition to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime. **Second consecutive SDK milestone to clear the
strong success bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`
§ 1.1 (R-56 anchor)** — strict separation from W8 on Phase 56
(+1.000 multi_service − corroboration on accuracy_full,
+1.000 vs FIFO and W7-2 too), backward-compat on Phase 55
(W9 ties W8 at 1.000 via the argmax-by-role-count gate),
backward-compat on Phase 54 (W9 ties W7-2 at 1.000),
no regression on Phase 53 synthetic (0.800), cross-bank stability
across 5/5 seeds, named falsifier regime (W9-4) correctly ties
FIFO at 0.000.  **First programme result whose strict-gain regime
is not solvable by the previous SDK's strongest method.***

### Added

- **Phase-56 multi-service-gold + cross-role-corroborated benchmark**
  (new): `vision_mvp/experiments/phase56_multi_service_corroboration.py`.
  Smallest deterministic regime where (a) every scenario has
  `gold_services` of size 2 (multi-service incident), (b) both gold
  services are corroborated by ≥ 2 distinct producer roles, (c) at
  least one decoy service has raw plurality but is corroborated by
  exactly 1 producer role. 5 base scenario builders × 2 replicates
  → 10-scenario default bank; named falsifier bank promotes a
  decoy to ≥ 2 distinct producer roles (W9-4 anchor).
- **`MultiServiceCorroborationAdmissionPolicy`** (new): in
  `vision_mvp/wevra/team_coord.py`. Deterministic, training-free
  admission rule that admits the **top-K cross-role-corroborated
  tier** (default `top_k=2, min_corroborated_roles=2`) via the
  argmax-by-role-count gate. Strictly generalises the SDK v3.9
  W8 single-tag corroboration policy (W9-3 backward-compat).
  Buffered factory `from_candidate_stream` is the W9-1 anchor.
  Re-exported as `TeamMultiServiceCorroborationAdmissionPolicy`.
- **`_dominant_tag_set`** helper (new): pure function with three
  structural properties (W9-2): single-role exclusion;
  argmax-tier collapse; argmax-tier multi-tag admission within
  `top_k` cap.
- **W9 theorem family** (new): W9-1 strict separation, W9-2
  argmax-tier strict-ordering, W9-3 backward-compat with W8
  + W7-2, W9-4 decoy-corroboration falsifier — all proved or
  proved-empirical on the pre-committed Phase-56 default. W9-C1
  / W9-C2 / W9-C3 conjectures (bundle-aware decoder, |gold|≥3,
  real-LLM transfer).
- **36 contract tests** in `test_wevra_multi_service_corroboration.py`:
  policy unit tests, bank shape, default config win, seed stability,
  falsifier behaviour, W9-3 backward-compat with Phase 55, audit
  invariance, cross-regime contract, public-API contract.
- **`docs/RESULTS_WEVRA_MULTI_SERVICE_CORROBORATION.md`** (new):
  milestone results note with W9 family theorem statements.
- **Frozen artefacts** in `docs/data/`: `phase56_multi_service_K4_n10.json`,
  `phase56_falsifier_K4_n10.json`, `phase56_seed_sweep.json`,
  `phase56_cross_regime.json`,
  `phase53_synthetic_w9_regression_check.json`.

### Changed

- **`SDK_VERSION`** bumped from `wevra.sdk.v3.9` to `wevra.sdk.v3.10`.
- **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** — bar anchor
  advanced to R-56; R-56 named regime added with mechanical-witness
  ingredient list; falsifying-failure list extended to gate W8-1
  contract test; canonical phrasing for SDK v3.10 added.
- **`docs/THEOREM_REGISTRY.md`** — W9-1/W9-2/W9-3/W9-4 + W9-C1/C2/C3
  added; W8-C1 marked DISCHARGED; date stamp v3.10.
- **`docs/RESEARCH_STATUS.md`** — ninth research axis (multi-service
  top-K corroboration) added; SDK v3.10 frontier section.
- **`docs/HOW_NOT_TO_OVERSTATE.md`** — W9 overstatement guards
  added (W9-1 conditionality, W8 multi-service-gold falsifier
  named, "we solved multi-agent context" still forbidden).
- **`docs/context_zero_master_plan.md`** — § 4.27 added (SDK v3.10
  milestone summary + post-v3.10 reading).
- **`docs/START_HERE.md`** — SDK v3.10 paragraph + W9 family summary;
  links to milestone result + success bar updated.

### Preserved

- **Wevra single-run product runtime contract.** Byte-for-byte
  unchanged from SDK v3.9. The Phase-45 product report schema
  (`PRODUCT_REPORT_SCHEMA = "phase45.product_report.v2"`) is
  unchanged.
- **SDK v3.5–v3.9 multi-agent surface.** Every fixed admission
  policy from previous SDKs (FIFO, priority, coverage,
  cohort_coherence, cross_role_corroboration) is unchanged; W7-2
  and W8-1 contract tests still pass byte-for-byte. The new W9
  policy is purely additive.
- **Lifecycle audit (T-1..T-7).** Holds on every cell of every
  regime (R-53 / R-54 / R-55 / R-56 default / R-56 falsifier).

## [3.9] — 2026-04-26 — SDK v3.9 — cross-role corroboration multi-agent coordination + W8 family

*Strictly additive on SDK v3.8. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``CrossRoleCorroborationAdmissionPolicy`` is a research-slice
addition to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime. **First SDK milestone to clear the strong success
bar of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1** —
strict separation from W7-2 on Phase 55 (+1.000 corroboration −
cohort_buffered, +1.000 corroboration − fifo on accuracy_full),
backward-compat on Phase 54 (corroboration ties W7-2 at 1.000),
no regression on Phase 53 synthetic (0.800) or 14B real-LLM
(0.800), cross-bank stability across 5/5 seeds, named falsifier
regime correctly ties FIFO at 0.000.*

### Added

- **`docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`** (new): pre-
  committed strong / partial / null success bars with named
  regimes (R-53 / R-54 / R-55).
- **Phase-55 decoy-plurality + cross-role-corroborated benchmark**
  (new): `vision_mvp/experiments/phase55_decoy_plurality.py`.
  Smallest deterministic regime where (a) decoy raw plurality
  breaks W7-2 single-tag plurality cohort coherence AND (b) gold
  cross-role corroboration provides a relational signal a (role,
  tag)-aggregating policy can exploit. Bench properties named and
  mechanically verified.
- **`CrossRoleCorroborationAdmissionPolicy`** (new): in
  `vision_mvp/wevra/team_coord.py`. Deterministic, training-free
  admission rule with score function `role_weight·|distinct_roles|
  + |raw_mentions|`. Buffered factory `from_candidate_stream` is
  the W8-1 anchor. Re-exported as
  `TeamCrossRoleCorroborationAdmissionPolicy`.
- **W8 theorem family**: W8-1 (strict separation, proved-empirical
  n=50), W8-2 (score-function strict ordering, proved structural),
  W8-3 (backward-compat with W7-2 on Phase 54, proved-empirical),
  W8-4 (decoy-corroboration falsifier, proved-empirical n=10).
  W8-C1 / W8-C2 / W8-C3 conjectures.
- **34 contract tests**:
  `vision_mvp/tests/test_wevra_cross_role_corroboration.py`.
- **Frozen reproducibility artefacts**:
  `docs/data/phase55_decoy_plurality_K4_n10.json` (default),
  `docs/data/phase55_falsifier_K4_n10.json` (W8-4),
  `docs/data/phase55_budget_sweep.json`,
  `docs/data/phase55_seed_sweep.json`,
  `docs/data/phase55_cross_regime.json`,
  `docs/data/phase53_real_llm_corroboration_check.json`.

### Changed

- `vision_mvp/wevra/__init__.py`: re-exports
  `TeamCrossRoleCorroborationAdmissionPolicy`; `SDK_VERSION`
  bumped to `"wevra.sdk.v3.9"`.
- `vision_mvp/tests/test_wevra_public_api.py`: SDK version test
  updated to v3.9; new corroboration export test.
- `docs/THEOREM_REGISTRY.md`: W8 family rows added; date stamp
  v3.9.
- `docs/RESEARCH_STATUS.md`: eighth research axis added.
- `docs/HOW_NOT_TO_OVERSTATE.md`: W8 overstatement guards added
  (W8-1 conditionality; "we solved multi-agent context" forbidden
  without naming the strong success bar; Phase-54/55 conflation
  forbidden; Phase-53/55 conflation forbidden).
- `docs/context_zero_master_plan.md`: § 4.26 SDK v3.9 added.
- `docs/START_HERE.md`: SDK v3.9 paragraph + canonical-reading
  pointer to the success-criterion doc.
- `docs/RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md`: new milestone
  results note.

### Honest scope

- The W8-1 win is **conditional** on the named bench property
  (decoy-plurality + cross-role-corroborated gold). The W8-4
  falsifier regime is the explicit named counterexample.
- Three named regimes is a stronger cross-regime result than two,
  but not "all regimes." Real production multi-agent teams have
  additional axes the W8 family does not test (heterogeneous
  producers, time-varying budgets, multi-round handoffs,
  multi-service gold). W8-C1 / W8-C2 / W8-C3 are conjectural;
  none yet shipped.
- The Wevra single-run product runtime contract is byte-for-byte
  unchanged from SDK v3.8. The new admission policy is a
  research-slice addition.

## [3.8] — 2026-04-26 — SDK v3.8 — cross-role cohort-coherence multi-agent coordination + W7 family

*Strictly additive on SDK v3.7. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new
``CohortCoherenceAdmissionPolicy`` is a research-slice addition
to the multi-agent coordination layer
(``vision_mvp.wevra.team_coord``), not part of the run-boundary
product runtime.*

### Added

- **Phase-54 cross-role cohort-coherence benchmark** (new):
  ``vision_mvp/experiments/phase54_cross_role_coherence.py``.
  Smallest deterministic multi-agent benchmark where cross-role
  coordination provides a strict structural advantage over
  substrate FIFO. Bench properties (gold-plurality, cross-role,
  budget-bound, decoder-pollution) are *named and mechanically
  verified* by the contract tests. Runs end-to-end without any
  LLM in the loop.
- **``CohortCoherenceAdmissionPolicy``** (new): in
  ``vision_mvp/wevra/team_coord.py``. Deterministic, training-free,
  interpretable cross-role admission rule. Two sub-modes:
  *streaming* (running cohort over already-admitted; arrival-
  order-sensitive) and *buffered* (pre-fitted plurality from the
  full candidate stream's payloads via
  ``from_candidate_payloads``; arrival-order-stable). Re-exported
  as ``TeamCohortCoherenceAdmissionPolicy``.
- **W7 theorem family**: W7-1 (FIFO unbeatability under low
  surplus, proved-empirical anchor on Phase-53), W7-1-aux
  (streaming cohort instability under arrival permutation,
  proved-empirical), W7-2 (cohort_buffered structural win under
  gold-plurality, proved-empirical, n=50 saturated, 5/5 stable
  across bank seeds), W7-2-conditional (K-sweep window,
  proved-empirical), W7-3 (extraction floor, proved-negative,
  Capsule Contract C5 corollary). W7-C1/C2/C3 conjectures cover
  multi-service-gold, decoder-side coordination, and real-LLM
  transfer extensions.
- **21 contract tests** for the new policy + bench:
  ``vision_mvp/tests/test_wevra_cross_role_coherence.py``.
- **Frozen reproducibility artefacts**:
  ``docs/data/phase54_cross_role_coherence_K4_n10.json`` (default
  config),
  ``docs/data/phase54_cross_role_coherence_budget_sweep.json``
  (K-sweep).
- **Milestone results note**:
  ``docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md``.

### Changed

- ``SDK_VERSION`` bumped to ``"wevra.sdk.v3.8"``.
- ``vision_mvp/wevra/__init__.py``: re-exports
  ``TeamCohortCoherenceAdmissionPolicy``.
- ``vision_mvp/wevra/team_coord.py``:
  ``ALL_FIXED_POLICY_NAMES`` extended with
  ``"cohort_coherence"``; new helper ``_candidate_service_tag``.
- ``docs/THEOREM_REGISTRY.md``: W7-1 / W7-1-aux / W7-2 /
  W7-2-conditional / W7-3 / W7-C1 / W7-C2 / W7-C3 rows added.
- ``docs/RESEARCH_STATUS.md``: seventh research axis added (W7
  family); now lists 7 coupled axes.
- ``docs/HOW_NOT_TO_OVERSTATE.md``: W7-overstatement guards
  added (cohort-coherence wins are *conditional* on bench
  properties; SDK v3.7 and SDK v3.8 results are both true,
  conditioned on different bench properties; *buffered* vs
  *streaming* distinction must be specified).
- ``docs/context_zero_master_plan.md``: § 4.25 added.
- ``docs/START_HERE.md``: SDK v3.8 paragraph added.

### Honest scope

- **The W7-2 win is conditional.** It depends on the bench having
  gold-plurality + foreign-service decoys + ``|candidates| >
  K_auditor``. The Phase-53 (real-LLM) reading is preserved
  exactly: substrate FIFO ties every fixed strategy at
  ``accuracy_full = 0.800`` because the bench has no surplus
  (W7-1).
- **The Wevra single-run product runtime contract is unchanged.**
  ``RunSpec`` / ``run`` / ``SweepSpec`` / ``run_sweep`` /
  report v2 schema: byte-for-byte identical from SDK v3.7.
- **The capsule layer's audit contribution is preserved.**
  T-1..T-7 hold on every Phase-54 cell unchanged.
- **Mac 2 still offline.** No two-Mac sharded inference happened
  in SDK v3.8; the ``MLXDistributedBackend`` integration boundary
  is byte-for-byte unchanged from SDK v3.6.

## [docs] — 2026-04-26 — documentation consolidation (no SDK change)

*Repo-cleanup only. No code change. SDK contract byte-for-byte
unchanged. Strictly additive on SDK v3.7.*

### Changed

- **Top-level Markdown clutter consolidated.** The repo root and
  `docs/` are reduced to a small canonical set; everything else is
  preserved under `docs/archive/`. The active scientific position is
  now obviously the live entry point and stale milestone notes can no
  longer read like current claims.
- **Canonical kept set** (top level): `README.md`, `ARCHITECTURE.md`,
  `CHANGELOG.md`, `LICENSE`, `CLAUDE.md`. **Canonical kept set**
  (`docs/`): `START_HERE.md`, `RESEARCH_STATUS.md`,
  `THEOREM_REGISTRY.md`, `HOW_NOT_TO_OVERSTATE.md`,
  `CAPSULE_FORMALISM.md`, `CAPSULE_TEAM_FORMALISM.md`,
  `context_zero_master_plan.md`, `MLX_DISTRIBUTED_RUNBOOK.md`,
  `RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` (latest milestone, kept live).
- **Archive layout** (`docs/archive/`):
  - `capsule-research/` — `RESULTS_CAPSULE_LEARNING.md` +
    `RESULTS_CAPSULE_RESEARCH_MILESTONE[1-6].md`.
  - `wevra-milestones/` — older Wevra milestone notes
    `RESULTS_WEVRA_{CAPSULE, CAPSULE_NATIVE, INTRA_CELL,
    DEEP_INTRA_CELL, INNER_LOOP, TEAM_COORD, DISTRIBUTED}.md`
    (SDK v3.0 → v3.6).
  - `pre-wevra-theory/` — pre-Wevra Context Zero theory volumes:
    `PROOFS.md`, `EXTENDED_MATH[_1-7].md`, `OPEN_QUESTIONS.md`,
    `FRAMEWORK.md`, `EVALUATION.md`, `MVP.md`, `ROADMAP.md`,
    `VISION_MILLIONS.md`, `MATH_AUDIT.md`,
    `HIERARCHICAL_DECOMPOSITION.md`, `WAVES.md`.
  - `legacy-progress-notes/` — sprint prompts, paradigm-shift
    summaries, the pre-Wevra benchmark-reproduction guide, the
    auto-generated theorem index.
- **`docs/archive/README.md`** *(new)* — archive index. Names every
  archived doc, points to the canonical replacement, and explains the
  read-only contract: the active scientific position is in `docs/`,
  the archive is historical record only.
- **Internal links updated.** Every cross-link inside the canonical
  docs (`README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/*.md`,
  `papers/*.md`) now resolves to the new file paths. Validated
  programmatically — zero broken Markdown links across the 14
  canonical docs.
- **`docs/START_HERE.md`** — adds a *Current canonical reading* table
  at the very top of the file. Mental-model diagram updated to show
  active vs archived theory paths.
- **`vision_mvp/scripts/generate_theorem_docs.py`** — auto-generated
  `THEOREMS_AUTO.md` now writes into
  `docs/archive/legacy-progress-notes/THEOREMS_AUTO.md` (was
  `docs/THEOREMS_AUTO.md`); the file was always a generated artefact,
  not a canonical claim source.

### Preserved

- All historical research material is intact under `docs/archive/`.
  No file deleted. No claim retracted. No theorem renumbered.
- `vision_mvp/RESULTS_PHASE*.md` (the per-phase research diary) is
  untouched — it lives with the code, not under `docs/`.
- The Wevra SDK public contract, the Capsule Contract C1..C6, and
  the W3 / W4 / W5 / W6 theorem families are unchanged.

## [SDK v3.7] — 2026-04-26 — model-scale vs capsule-structure on multi-agent coordination (Phase-53 + W6 family)

*Strictly additive on SDK v3.6. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new surface is
the Phase-53 stronger-model multi-agent benchmark + W6 theorem
family. Mac 2 is still offline; no two-Mac sharded inference
happened in this milestone — the ``MLXDistributedBackend``
integration boundary is byte-for-byte unchanged from SDK v3.6
and waits for the runbook.*

### Added

- **`vision_mvp/experiments/phase53_scale_vs_structure.py`** *(new)* —
  Phase-53 stronger-model multi-agent benchmark. Drives the team
  coordinator with a real-LLM producer-role extractor across
  three model regimes (synthetic / qwen2.5:14b-32k / qwen3.5:35b)
  × five admission strategies (substrate, capsule_fifo,
  capsule_priority, capsule_coverage, capsule_learned) on the
  same candidate-handoff stream. Reports a clean ``model regime ×
  admission strategy`` decomposition with cross-regime
  candidate-kind TVD.
- **`vision_mvp/tests/test_wevra_scale_vs_structure.py`** *(new)*
  — 19 contract tests: parser robustness on the closed-vocabulary
  claim grammar (16 cases), backend duck-typing, audit_ok grid
  end-to-end with a deterministic stub backend, schema lock.
- **`docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`** *(new)* — full
  milestone results note. Theorem-forward; declares W6-1..W6-4
  proved-or-empirical and W6-C1..W6-C5 conjectures (W6-C1 / C2
  drafted-then-falsified, W6-C3 positive, W6-C4 / C5 new).
- **`docs/data/phase53_scale_vs_structure_K4_n5.json`** *(new
  artefact)* — frozen benchmark output for reproducibility.

### Changed

- **`vision_mvp/wevra/__init__.py`** — `SDK_VERSION` bumped to
  `wevra.sdk.v3.7`. No public API change.
- **`docs/THEOREM_REGISTRY.md`** — W6-1 / W6-2 / W6-3 / W6-4 +
  W6-C1 / W6-C2 / W6-C3 / W6-C4 / W6-C5 rows added. The W4-C1
  row (SDK v3.5 conjecture) is now annotated as **conditional**:
  empirical-positive on its anchor distribution; falsified
  out-of-distribution on the Phase-53 real-LLM regime
  (capsule_learned 0.4 vs fixed 0.8 on synthetic and qwen2.5:14b;
  ties at qwen3.5:35b at 0.8/0.8).
- **`docs/RESEARCH_STATUS.md`** — sixth research axis added;
  active-conjectures section refreshed with W6-C family.
- **`docs/context_zero_master_plan.md`** — § 4.24 added: full
  Phase-53 narrative, W6 / W6-C summary, W4-C1 conditional
  reading, honest scope (Mac 2 offline, single-Mac qwen3.5:35b
  is the strongest model class actually exercised).
- **`docs/START_HERE.md`** — headline paragraph updated to
  reference the SDK v3.7 result and the *audit-axis* tightening
  of the original Context-Zero thesis.

### Headline empirical result

(n=5 saturated, K_auditor=4, T_auditor=128, three model regimes,
deterministic seeds (31, 32, 33))

| regime           | substrate | fixed capsule | learned |
| ---------------- | --------- | ------------- | ------- |
| synthetic        | 0.800     | 0.800         | 0.400   |
| qwen2.5:14b-32k  | 0.800     | 0.800         | 0.400   |
| qwen3.5:35b      | 0.800     | 0.800         | 0.800   |

* `structure_gain[regime]` = -0.4 / -0.4 / 0.0 (non-positive
  everywhere; scale narrows a *deficit*, not a *surplus*).
* `scale_gain[capsule_learned]` = +0.4; `scale_gain[fixed]` = 0.0.
* Cross-(14B, 35B) candidate-kind TVD = 0.167.
* Capsule-team lifecycle audit ``audit_team_lifecycle.is_ok()``
  = 60/60 across (regime × capsule strategy × scenario).

### Theorem registry deltas

- **W6-1 (proved + mechanically-checked).** Lifecycle audit
  T-1..T-7 holds 60/60 across the Phase-53 grid.
- **W6-2 (proved).** Phase-53 driver accepts duck-typed
  ``LLMBackend``.
- **W6-3 (proved + mechanically-checked).** Parser robustness
  on the closed-vocabulary claim grammar.
- **W6-4 (proved-empirical, real LLM, n=5 saturated).** The
  ``accuracy_full`` / ``structure_gain`` / ``scale_gain``
  decomposition is what is reported above.
- **W6-C1, W6-C2 (drafted, FALSIFIED-empirical).** Structure-
  preservation under scale (W6-C1) and synthetic→real-LLM
  transfer of the learned admission scorer (W6-C2) are both
  falsified on Phase-53 default; honest revised reading is in
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` § 4.3.
- **W6-C3 (empirical-positive).** Cross-(14B, 35B) candidate-
  kind TVD = 0.167 > 0.10 falsifier.
- **W6-C4, W6-C5 (new conjectures).** Substrate-FIFO competitive-
  ness at sufficient K, and scale-narrows-the-OOD-gap of the
  per-role admission scorer.

### Honest scope

* Mac 2 (192.168.12.248) is offline at the time of this
  milestone (ARP "incomplete"). **No two-Mac sharded inference
  ran.** No 70 B-class model ran. The strongest model class
  exercised is **single-Mac** qwen3.5:35b (36 B-MoE Q4) via
  Mac 1 Ollama.
* The MLX-distributed integration boundary
  (``MLXDistributedBackend``) is byte-for-byte unchanged from
  SDK v3.6 and remains correct against the in-process stub
  (W5-3). The runbook (`docs/MLX_DISTRIBUTED_RUNBOOK.md`) is the
  operator path when Mac 2 returns.
* Phase-53 is **incident-triage-bench-internal**. External
  validity to other multi-agent benchmarks is open
  (`task_scale_swe.py`, `phase33_security_escalation.py` are
  obvious next targets).
* The W4-C1 (SDK v3.5) reading on its anchor config (Phase-52
  default, K=8, spurious=0.30) is unchanged. The new SDK v3.7
  reading is OOD.

### Tests + validation

* `python3 -m unittest -v vision_mvp.tests.test_wevra_scale_vs_structure`
  → **19 tests pass in 0.069 s**.
* `python3 -m unittest vision_mvp.tests.test_wevra_team_coord
  vision_mvp.tests.test_wevra_llm_backend
  vision_mvp.tests.test_wevra_capsule_native_inner_loop
  vision_mvp.tests.test_wevra_capsule_native
  vision_mvp.tests.test_wevra_capsule_native_intra_cell
  vision_mvp.tests.test_wevra_capsule_native_deeper
  vision_mvp.tests.test_wevra_scale_vs_structure`
  → **116 tests pass in 3.207 s** (SDK v3.6 invariants intact).
* `python3 -m vision_mvp.experiments.phase53_scale_vs_structure
  --endpoint http://192.168.12.191:11434
  --models synthetic,qwen2.5:14b-32k,qwen3.5:35b
  --n-eval 5 --K-auditor 4 --T-auditor 128
  --out /tmp/wevra-distributed/phase53_scale_vs_structure_K4.json`
  → 14B LLM wall 92.6 s; 35B LLM wall 152.0 s; n_results = 75.
  Frozen at `docs/data/phase53_scale_vs_structure_K4_n5.json`.

## [SDK v3.5] — 2026-04-26 — capsule-native multi-agent team coordination (research slice)

*Strictly additive on SDK v3.4. The Wevra single-run product
runtime contract is byte-for-byte unchanged. The new surface is a
capsule-native multi-agent coordination research slice that
runs side-by-side with the Wevra SDK.*

### Added

- **Three new closed-vocabulary `CapsuleKind` values** — `TEAM_HANDOFF`
  (capsule-native multi-agent handoff; distinct from `HANDOFF`
  which adapts a substrate `TypedHandoff`), `ROLE_VIEW` (per-role
  admitted view of one coordination round; `max_parents = K_role`,
  `max_tokens = T_role`), `TEAM_DECISION` (team-level decision).
- **`vision_mvp.wevra.team_coord`** — `RoleBudget`,
  `DEFAULT_ROLE_BUDGETS`, `capsule_team_handoff`,
  `capsule_role_view`, `capsule_team_decision`, three fixed
  admission policies (`FifoAdmissionPolicy`,
  `ClaimPriorityAdmissionPolicy`, `CoverageGuidedAdmissionPolicy`),
  `TeamCoordinator`, `audit_team_lifecycle` over invariants
  `T-1..T-7` (Theorem **W4-1**, *proved + mechanically-checked*).
- **`vision_mvp.wevra.team_policy`** —
  `LearnedTeamAdmissionPolicy` (per-role logistic-regression
  scorer over six capsule features), `TrainSample`, `TrainStats`,
  `train_team_admission_policy`. Numpy-only; deterministic given
  seed.
- **`vision_mvp/experiments/phase52_team_coord.py`** — reference
  benchmark over a noisy-extraction expansion of the Phase-31
  incident-triage bank. Cross-seed result on default config
  ($K_\text{auditor}=8$, $T_\text{auditor}=256$,
  $n_\text{eval}=31$, ``train_seed ∈ {0, …, 11}``,
  ``PYTHONHASHSEED=0``): **learned policy** admits **strictly
  fewer handoffs** than the strongest fixed baseline
  (coverage-guided) on every train seed (12/12), with mean
  savings ≈ 1.26 handoffs per scenario. The learned policy also
  improves pooled team-decision accuracy on most train seeds
  (gap on `accuracy_full` > 0 in 11/12 seeds, mean **+0.054**;
  gap on `accuracy_root_cause` > 0 in 8/12 seeds, mean
  **+0.032**) — but the accuracy advantage **reverses at higher
  noise** (`spurious_prob = 0.50`). `audit_ok_rate = 1.000` for
  every capsule strategy on every seed. Conjecture **W4-C1**:
  budget-efficiency dominance is robust per-seed; accuracy
  advantage is mean-positive on the default noise config but
  not strict per-seed; advantage does not survive heavier
  noise. (See ``docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md`` § Cross-seed
  result for the canonical reading; ``docs/HOW_NOT_TO_OVERSTATE.md``
  forbids reporting single-seed numbers without the cross-seed
  distribution.)
- **Theorems** — W4-1 (proved + mechanically-checked); W4-2
  (proved-conditional: coverage-implies-correctness); W4-3
  (proved-negative: per-role budget below the role's causal-
  share floor cannot be rescued by *any* admission policy).
- **Conjectures** — W4-C1, W4-C2 (cohort-lifted role view closes
  W4-3 sub-class), W4-C3 (capsule admission rule subsumes
  Phase-36 adaptive-sub).
- **`docs/CAPSULE_TEAM_FORMALISM.md`** — formal model.
- **`docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md`** — milestone note.
- **`vision_mvp/tests/test_wevra_team_coord.py`** — 22 contract
  tests.
- **README**, **START_HERE**, **RESEARCH_STATUS**,
  **THEOREM_REGISTRY**, **HOW_NOT_TO_OVERSTATE**, **master plan
  §4.22** — all updated.

### Compatibility

- All 85 capsule-native run-boundary tests (v3.1..v3.4) +
  Phase-31 typed-handoff tests continue to pass byte-for-byte.
  Team-layer tests are 22 additional contracts.
- The Wevra `wevra` console scripts are unchanged. The team layer
  ships as `vision_mvp.wevra.team_coord` /
  `vision_mvp.wevra.team_policy` and is also re-exported from
  the top-level `vision_mvp.wevra` namespace as
  `TeamCoordinator`, `audit_team_lifecycle`,
  `LearnedTeamAdmissionPolicy`, etc.

### Honest scope

The Phase-52 benchmark is synthetic; the result *direction* is
robust under deterministic noise; cross-bench transfer is open.
"We solved multi-agent context" is **forbidden** by
`docs/HOW_NOT_TO_OVERSTATE.md`; the defensible reading is
W4-1 / W4-2 / W4-3 / W4-C1 above.

## [SDK v3.4] — 2026-04-26 — sub-sub-intra-cell PROMPT / LLM_RESPONSE slice + synthetic mode + cross-model parser-boundary research

*Strictly additive on SDK v3.3. Every v3.3 contract test (18) still
passes byte-for-byte; capsule view schema name unchanged
(`wevra.capsule_view.v1` — PROMPT / LLM_RESPONSE payloads are
additive). Full Wevra + capsule test suite green (199 tests).*

### Added
- **PROMPT capsule kind** (parent: SWEEP_SPEC; Theorem W3-42).
  Records prompt SHA-256 + byte length + bounded text snippet
  (≤ 4 KiB) + model_tag + prompt_style + coordinates.
  Idempotent on content (Capsule Contract C1) — byte-identical
  prompts collapse to one capsule.
- **LLM_RESPONSE capsule kind** (parent: PROMPT; Theorem W3-43).
  Records response SHA-256 + byte length + bounded snippet +
  elapsed milliseconds + coordinates. Admission rejects if
  prompt CID is not yet sealed (Capsule Contract C5).
- **`CapsuleNativeRunContext.seal_prompt`** /
  **`seal_llm_response`** runtime methods, plus
  **`seal_parse_outcome(llm_response_cid=...)`** optional
  argument. The end-to-end inner-loop chain is now five typed
  capsules: `PROMPT → LLM_RESPONSE → PARSE_OUTCOME →
  PATCH_PROPOSAL → TEST_VERDICT`.
- **`capsule_from_prompt`**, **`capsule_from_llm_response`**
  adapters; `PROMPT_TEXT_CAP` / `LLM_RESPONSE_TEXT_CAP` constants.
- **Lifecycle audit invariants L-9 / L-10 / L-11** (Theorems
  W3-44 / W3-45):
  - L-9: PROMPT.parents == (SWEEP_SPEC,).
  - L-10: LLM_RESPONSE has exactly one parent, a sealed PROMPT.
  - L-11: PARSE_OUTCOME / LLM_RESPONSE coordinate consistency
    (instance_id / parser_mode / apply_mode / n_distractors;
    strategy may differ).
- **Synthetic-LLM mode**: `SweepSpec(mode="synthetic",
  synthetic_model_tag=<tag>)`. Uses a deterministic in-process
  `SyntheticLLMClient` instead of an Ollama endpoint. Seven
  calibrated distributions ship in
  `vision_mvp.wevra.synthetic_llm.SYNTHETIC_MODEL_PROFILES`:
  `clean`, `unclosed`, `prose`, `empty`, `fenced`,
  `multi_block`, `mixed`. The full PROMPT / LLM_RESPONSE /
  PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT chain seals
  end-to-end without network access.
- **Cross-model parser-boundary experiment** (Conjecture W3-C6,
  empirical):
  `vision_mvp.experiments.parser_boundary_cross_model`. Sweeps
  `(model_tag, parser_mode)` across the synthetic distribution
  library; reports cross-distribution PARSE_OUTCOME failure-kind
  TVD up to 1.000 and parser-mode (strict→robust) shift up to
  1.000 on `synthetic.unclosed`. Reproducible from CLI:
  `python3 -m vision_mvp.experiments.parser_boundary_cross_model`.
- **16 new contract tests** in
  `vision_mvp/tests/test_wevra_capsule_native_inner_loop.py`
  covering W3-42 / W3-43 / W3-44 / W3-45 / W3-C6.

### Changed
- **`SDK_VERSION`** bumped to `wevra.sdk.v3.4`.
- **`CapsuleKind.ALL`** now includes `PROMPT` and `LLM_RESPONSE`.
- **`render_view.payload_kinds_always`** extended to include
  PROMPT and LLM_RESPONSE (so on-disk audits can navigate the
  full inner-loop chain from `capsule_view.json` alone).
- **`CapsuleLifecycleAudit.RULES`** extended from 8 rules to 11.
- **W3-13** (DAG height ≤ 4 on canonical run pattern) is updated
  to ≤ 5 on canonical SDK v3.4 runs (the inner-loop chain adds
  one structural layer). Documented in
  `docs/CAPSULE_FORMALISM.md` § 4.J.
- **Conjecture W3-C5 (legacy SDK v3.3)** is **DISCHARGED** by
  Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **Conjecture W3-C4 (legacy SDK v3.3)** is **superseded** by the
  sharper synthetic reading W3-C6.

### Documentation
- New milestone note: **`docs/archive/wevra-milestones/RESULTS_WEVRA_INNER_LOOP.md`**.
- `docs/CAPSULE_FORMALISM.md` § 4.J added (W3-42 / W3-43 / W3-44 /
  W3-45 / W3-C6 + W3-C5-discharged).
- `docs/THEOREM_REGISTRY.md`, `docs/RESEARCH_STATUS.md`,
  `docs/HOW_NOT_TO_OVERSTATE.md` updated for SDK v3.4.
- `docs/START_HERE.md` adds "What changed in SDK v3.4" section.
- `docs/context_zero_master_plan.md` § 4.21 added.
- `papers/wevra_capsule_native_runtime.md` strengthened —
  capsule-native execution is now its real centre, with strict
  claim taxonomy covering PROMPT / LLM_RESPONSE chain and the
  W3-C6 empirical anchor.
- README headline + stability matrix updated.

## [0.5.1] — 2026-04-22 — Wevra identity & clarity pass

*Documentation / exemplar milestone. No SDK-contract change; all 1349
Slice-2 tests still pass.*

### Added
- **`docs/START_HERE.md`** — canonical one-pass orientation for new
  readers. Classifies every top-level surface (Wevra SDK, CLI,
  extension protocols, unified runtime, legacy product path, core
  substrate, research shards, boundary). Meant to be the answer to
  "what is this repo?" without duplicating the README or the master
  plan.
- **`examples/out_of_tree_plugin/wevra-markdown-sink/`** — first
  in-repo exemplar of a standalone pip-installable Wevra plugin
  package. Declares `[project.entry-points."wevra.report_sinks"]`,
  registers a Markdown `ReportSink` via
  `importlib.metadata.entry_points`, and requires zero edit under
  `vision_mvp/`. Closes master-plan § 10.5 ledger item 2 at the
  machinery-plus-artifact level (only the "published by a third
  party" condition remains future).
- **`vision_mvp/RESULTS_WEVRA_IDENTITY.md`** — theory-forward results
  note with theorem-style claims (W-IDN-1 identity projection,
  W-IDN-2 orientation sufficiency, W-IDN-3 extension-surface
  reality) and three conjectures (W-IDN-C1 cold-agent
  classification, W-IDN-C2 stable-identity robustness, W-IDN-C3
  distinctiveness via composition rather than primitive novelty).

### Changed
- **README headline** now leads with **Wevra** (the shipped product)
  and positions CASR as original-substrate research; the scaling
  claims are preserved and re-anchored to Theorem 3 in `docs/archive/pre-wevra-theory/PROOFS.md`.
- **ARCHITECTURE.md headline** re-anchored to Wevra + Context Zero;
  a framing callout was added before the Phase 26–44 block so
  readers know that block is a historical incremental record and
  the durable architecture is the layered substrate diagram + § 3
  of the master plan.
- **`vision_mvp/__init__.py`** top-level docstring: Wevra is the
  shipped product; `CASRRouter` is explicitly research-grade code
  used by the SDK under the hood.
- **`vision_mvp/api.py`** `CASRRouter` docstring no longer says
  "Phase-3 hierarchical protocol" or "CASR-theoretic optimum" in
  places where a user would read them as current product contract;
  the O(log N) bound is now anchored to Theorem 3.
- **`vision_mvp/product/__init__.py`** retitled from "Phase-45
  product-grade orchestration surface" to "Legacy product modules
  (pre-Wevra import path)" — same code, correct framing.
- **`pyproject.toml`** — clearer comment on the `casr` legacy
  script; public CLI stays `wevra` / `wevra-import` / `wevra-ci`.
- **Master plan § 10** — short "Programme vs Product" callout near
  the top; § 10.1 stability matrix row for out-of-tree plugins
  updated from "boundary / next-slice" to "exemplar landed";
  § 10.3 B.6 note and § 10.5 ledger item 2 updated.

### Not changed (deliberately)
- The Wevra SDK contract (every Slice 2 public symbol remains).
- Any test; suite is green at 1349/1349.
- Docker-first-by-default flip for untrusted JSONLs (still Slice 3).
- GitHub Actions release-on-real-tag firing (workflow still declared,
  not yet exercised on a real tag).

## [0.5.0] — 2026-04-22 — Wevra SDK Slice 2

### Added
- **Extension system** (`vision_mvp/wevra/extensions/`). Three
  runtime-checkable Protocols — `SandboxBackend`, `TaskBankLoader`,
  `ReportSink` — each with an in-process registry and discovery via
  `importlib.metadata.entry_points` under groups
  `wevra.sandboxes`, `wevra.task_banks`, `wevra.report_sinks`.
  One worked example (`JsonlWithMetaSink`) and a contract test
  suite that exercises the full register→resolve→emit path.
- **Unified mock/real runtime** (`vision_mvp/wevra/runtime.py`).
  New `SweepSpec` dataclass; single `run_sweep(spec)` entry point
  dispatches mock and real runs through the same substrate
  primitives. Real runs execute in-process when
  `RunSpec.acknowledge_heavy=True`; otherwise the SDK refuses to
  start the heavy run and emits the resolved launch command.
- **`RunSpec.acknowledge_heavy`** and **`RunSpec.report_sinks`** —
  first-class cost gate and plugin hook on the top-level SDK spec.
- **`HeavyRunNotAcknowledged`** exception — strict cost-gate signal.
- **Env-driven endpoints**: `WEVRA_OLLAMA_URL_MAC1`,
  `WEVRA_OLLAMA_URL_MAC2`, `WEVRA_OLLAMA_URL` override profile-
  declared URLs at runtime. No hard-coded cluster IP is baked into
  code paths that a third-party consumer has to edit.
- **`--acknowledge-heavy` / `--report-sink`** flags on `wevra`.
- **Report schema bump**: `phase45.product_report.v2`. v1 remains
  accepted by `wevra-ci`; both listed in `EXPECTED_REPORT_SCHEMAS`.
- **GitHub Actions workflow** (`.github/workflows/wevra-ci.yml`):
  SDK contract tests on 3.10/3.11/3.12, console-script smoke,
  `python -m build` sdist+wheel, release on tag.
- **Cluster-backed validation artifact** under
  `vision_mvp/artifacts/wevra_slice2_g1/` — real ASPEN `mac1`
  `qwen2.5-coder:14b` run launched via `wevra.run(RunSpec(...,
  acknowledge_heavy=True))`, with provenance manifest and
  `wevra-ci` verdict.
- **Theory note**: `vision_mvp/RESULTS_WEVRA_SLICE2.md` —
  theorem-style claims W2-1 … W2-4.

### Changed
- `SDK_VERSION` bumped to `wevra.sdk.v2`. The bump is additive;
  every Slice 1 public symbol remains available.
- `CI gate` accepts v1 and v2 report schemas.
- `product/runner.py` now routes all sweeps through
  `wevra.runtime.run_sweep` instead of the legacy
  `_real_sweep_stub`.

### Deprecated
- `_real_sweep_stub` / `_mock_sweep` in `vision_mvp/product/runner.py`
  are private and will be removed in a future release; external code
  should use `wevra.run_sweep(SweepSpec(...))`.

### Next-slice (deferred, still honest)
- Docker-first sandbox as the default for public/untrusted JSONLs
  (backend exists; default-flip is Slice 3).
- Public SWE-bench-Lite JSONL on local disk (🧱 external).
- Resident ≥70B coder-finetuned model (🧱 external).

## [0.4.0] — 2026-04-21 — Wevra SDK Slice 1

See `docs/context_zero_master_plan.md` § 10.2.

- Introduced `vision_mvp/wevra/` stable SDK boundary.
- `RunSpec` / `run`, `WevraConfig`, `build_manifest`, schema
  constants, profile/report/ci_gate/import_data re-exports.
- Provenance manifest (`wevra.provenance.v1`) on every run.
- Console scripts: `wevra`, `wevra-import`, `wevra-ci`.
- Package renamed to `wevra` on PyPI; `SDK_VERSION = wevra.sdk.v1`.
- `sys.path.insert` hacks removed from product modules.
- Contract tests: `test_wevra_public_api.py`, `test_wevra_provenance.py`.

---

## [0.1.0] — 2026-04-16

Initial alpha release. One continuous research session.

### Added — Core library (`vision_mvp/`)

- **`CASRRouter`** — black-box public API. `step(observations) -> estimates`.
- Core primitives: `Bus`, `Agent`, `Manifold` (given basis),
  `StreamingPCA` (learned basis), `Stigmergy` (CRDT register),
  `Workspace` (top-k admission), `NeuralPredictor` and `PredictorBank`
  (vectorized across agents).
- Phase-6 additions: `MarketWorkspace` (VCG pricing),
  `SharedRNG`/`DeltaChannel` (pre-shared randomness), `AdaptiveScale` and
  `ContinuousScaleProjector` (continuous-scale projection).
- Six coordination protocols: `naive`, `gossip`, `manifold_only`,
  `full_stack`, `adaptive`, `hierarchical`, `holographic`, `swarm`, and
  `llm_protocols` (real LLM agents via Ollama).
- Two coordination tasks: `consensus` (static) and
  `drifting_consensus` (non-stationary with optional shock).

### Added — Experiments & results

- Phase 1 through Phase 5 runnable experiment harnesses under
  `vision_mvp/experiments/`.
- Measured scaling law: peak per-agent context = ⌈log₂ N⌉ exactly at
  every N ∈ {10, 50, 200, 1 000, 5 000, 10 000, 20 000, 50 000, 100 000}.
- Real LLM demonstration at N = 10 (local qwen2.5:0.5b via Ollama) showing
  34 % token savings with 100 % accuracy.

### Added — Theory

- **`docs/archive/pre-wevra-theory/PROOFS.md`** — twelve formal theorems, each with a proof and a
  machine-checkable empirical counterpart in `tests/`.
- **`EXTENDED_MATH_[1–7].md`** — 72-framework survey converging on the
  O(log N) bound from Information Bottleneck through Geometric Langlands.
- **`docs/archive/pre-wevra-theory/VISION_MILLIONS.md`** — the 10-idea paradigm shift for million-agent
  systems. 6 of 10 ideas implemented.

### Added — Tests

- **94 tests**, all passing (0.45 s total wall time):
  - 55 core-module unit tests.
  - 15 protocol integration & regression tests (including the scaling-law
    assertion `test_full_stack_peak_context_is_log_n`).
  - 13 Phase-6 tests (market, shared randomness, continuous scale).
  - 11 public-API (`CASRRouter`) tests.

### Added — Developer UX

- `pyproject.toml` (installable as `context-zero`).
- `LICENSE` (MIT), `.gitignore`, `CHANGELOG.md`, top-level `README.md`.
- `casr` CLI entry-point (`python -m vision_mvp demo|scale|phase|test|info`).
- Four runnable `examples/`:
  1. basic consensus
  2. drift tracking
  3. scaling demo
  4. local LLM coordination

### Not yet

- Real LLM tests at N > 10 (need bigger compute budget).
- Async variants (current protocol is synchronous).
- A formal peer-review cycle for the math.
- PyPI upload.

All the mathematics says O(log N). The code and the test suite confirm it.
The next step is to run it in anger on harder tasks and let skeptical
reviewers tear it apart.
