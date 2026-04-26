# Research status — canonical, current

> Single-source-of-truth for the *active* research position of the
> Context Zero programme. If this file disagrees with any other
> doc on what is *true now*, this file is right and the other file
> is stale. For *theorem-by-theorem* status, see
> `docs/THEOREM_REGISTRY.md`. For *what may be claimed*, see
> `docs/HOW_NOT_TO_OVERSTATE.md`. Last touched: SDK v3.7,
> 2026-04-26.

## TL;DR

The programme now has **six** coupled research axes, each with a
sharp status:

1. **Capsule contract / runtime** — *active, advancing*. The
   contract (C1..C6) is settled. SDK v3.4 pushes capsule-native
   execution to the LLM byte boundary inside one Wevra run
   (W3-42..W3-45). The lifecycle audit covers L-1..L-11.
2. **Multi-agent capsule coordination** — *active, new (SDK
   v3.5)*. Capsule-native team coordination via TEAM_HANDOFF /
   ROLE_VIEW / TEAM_DECISION capsules. ``TeamCoordinator`` drives
   one round; ``audit_team_lifecycle`` mechanically verifies
   invariants T-1..T-7 (Theorem W4-1). Coverage-implies-
   correctness (W4-2) and local-view limitation (W4-3) hold on
   the Phase-52 incident-triage bench. A learned per-role
   admission policy admits **strictly fewer handoffs** (12/12
   train seeds, deterministic in direction) and improves pooled
   team-decision accuracy *on most seeds* (gap_full > 0 in 11/12
   seeds, mean +0.054; gap_root_cause > 0 in 8/12 seeds, mean
   +0.032) over the strongest fixed baseline (coverage-guided)
   on the Phase-52 default config — but the accuracy advantage
   reverses at higher noise (W4-C1 honest reading). This is the
   team-level slice of the original Context-Zero "solve context
   for multi-agent teams" thesis — the first slice that runs the
   capsule abstraction *between* agents, not just inside one run.
3. **Decoder frontier** — *open, with sharp limitation theorems*.
   The strict pre-Phase-50 paradigm-shift bar (W3-C7 strict) is
   **retracted** (W3-26, W3-27). The defensible reading is
   W3-C9 (Phase-49 candidate at $n=80$, gap reading at zero-shot).
   The next research direction is the relational decoder at
   higher level (Phase 51, W3-30 / W3-31 / W3-C10).
4. **Substrate primitives** — *settled*. CASR routing, exact
   memory, typed handoffs, escalation threads, adaptive
   subscriptions. ~1500 substrate tests, no active development on
   substrate primitives themselves.
5. **Two-Mac distributed inference + real cross-LLM measurement**
   — *active, settled (SDK v3.6)*. The chosen path for one-larger-
   model inference across two Apple Silicon Macs is **MLX
   distributed** (under `mpirun mlx_lm.server`); the Wevra-side
   integration boundary is one duck-typed `LLMBackend` Protocol
   plus an `MLXDistributedBackend` adapter that talks
   OpenAI-compatible HTTP. Real cross-LLM measurement on the
   available model class (Qwen-2.5-14B-dense vs
   Qwen-3.5-35B-MoE on Mac 1) yields **W5-1
   (proved-empirical)**: cross-model PARSE_OUTCOME failure-kind
   TVD = 1.000 under strict parsing on the bundled bank,
   collapsing to 0.000 under robust parsing — the **first real
   confirmation** that the capsule-native runtime survives a
   2.4× model-class jump and a dense → MoE architecture swap
   without spine modification. The two-Mac MLX-distributed path
   is **experimental infrastructure**, not product; the Wevra
   single-run product runtime contract is byte-for-byte
   unchanged. Mac 2 remains offline at the time of SDK v3.7
   (192.168.12.248 ARP "incomplete"); the runbook is the
   operator path when Mac 2 returns.
6. **Model-scale vs capsule-structure on multi-agent
   coordination** — *active, new (SDK v3.7)*. **Phase-53**
   benchmark (`vision_mvp/experiments/phase53_scale_vs_structure.py`)
   replaces the Phase-52 deterministic producer-role extractor
   with a real-LLM extractor and decomposes ``accuracy_full``
   across (model regime × admission strategy). Headline (n=5,
   K_auditor=4): every fixed admission strategy (substrate /
   capsule_fifo / capsule_priority / capsule_coverage) achieves
   ``accuracy_full = 0.800`` in every model regime; only
   ``capsule_learned`` varies (0.400 on synthetic and 14B; 0.800
   on 35B). ``structure_gain`` is **non-positive at every
   regime** (-0.4 / -0.4 / 0.0); ``scale_gain[capsule_learned]
   = +0.4``, ``scale_gain[fixed] = 0.0``. **W4-C1 is conditionally
   falsified** out-of-distribution on the real-LLM regime
   (capsule_learned underperforms FIFO by 0.4 on synthetic and
   14B; ties at 35B). Honest reading: scale closes a *structure
   deficit* (created by OOD over-rejection of clean candidates
   by the SDK v3.5 learned policy), not a *structure surplus*.
   The capsule layer's load-bearing contribution at this
   benchmark is the **lifecycle audit (T-1..T-7, 60/60 across
   regimes)**, not admission policy gains. The W6 family
   (W6-1/2/3/4 proved + mechanically-checked + empirically-
   saturated) anchors the milestone formally; the W6-C family
   (W6-C1/C2 falsified-empirical, W6-C3 positive, W6-C4/C5
   conjectural) makes the empirical reading falsifiable.

## Current frontier (SDK v3.7, 2026-04-26)

### Active moves (SDK v3.7 — stronger-model multi-agent benchmark + W6 family)

- **Phase-53 stronger-model multi-agent benchmark.**
  ``vision_mvp.experiments.phase53_scale_vs_structure`` runs
  three model regimes (synthetic / qwen2.5:14b-32k /
  qwen3.5:35b) × four capsule admission strategies + the
  Phase-31 substrate baseline on the same candidate-handoff
  stream. Real LLM calls hit Mac 1 Ollama at
  ``192.168.12.191:11434`` (Mac 2 still offline). Wall: 14B =
  92.6 s, 35B = 152 s.
- **Theorem family W6.** W6-1 (audit-OK grid 60/60),
  W6-2 (backend duck-typing), W6-3 (parser robustness on the
  closed-vocabulary claim grammar) are proved + mechanically-
  checked. W6-4 (the empirical decomposition) is proved-empirical
  on n=5 saturated.
- **Conditional falsification of W4-C1.** The SDK v3.5
  learned-admission-policy advantage **does not transfer
  out-of-distribution** to the real-LLM regime. Per-regime gap
  (capsule_learned − capsule_fifo): -0.4 (synthetic) / -0.4
  (qwen2.5:14b-32k) / 0.0 (qwen3.5:35b). The W4-C1 row in the
  registry is now conditional (see § 4.4 of
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`).
- **Honest scope.** Mac 2 is still offline; no two-Mac sharded
  inference run happened in SDK v3.7. The strongest model class
  exercised is single-Mac qwen3.5:35b (36 B-MoE). The
  ``MLXDistributedBackend`` adapter is unchanged from SDK v3.6
  and remains correct against the in-process stub.

### Active conjectures (SDK v3.7)

- **W6-C1**: drafted-conjecture-falsified — structure_gain is
  non-positive at every regime tested on Phase-53 default;
  scale narrows a deficit (not a surplus).
- **W6-C2**: drafted-conjecture-falsified — synthetic→real
  transfer of the learned admission scorer LOSES to FIFO out-
  of-distribution.
- **W6-C3**: empirical-positive — cross-(14B, 35B) candidate-
  kind TVD = 0.167 on the pooled (source_role × claim_kind)
  histogram (above the 0.10 falsifier).
- **W6-C4**: new conjectural-empirical — substrate FIFO is
  competitive with every capsule admission policy at sufficient
  K_auditor; falsifier search direction is K_auditor ∈ {1, 2, 3}.
- **W6-C5**: new conjectural-empirical — model scale narrows
  the OOD generalisation gap of the per-role admission scorer
  trained on synthetic noise.

## Previous frontier (SDK v3.6, 2026-04-26)

### Active moves (SDK v3.6 — two-Mac distributed inference + real cross-LLM)

- **Chosen two-Mac inference path: MLX distributed.** Apple-
  official, supports Llama / Qwen / Mistral natively, and
  exposes a single OpenAI-compatible HTTP server (head rank)
  regardless of single-host or sharded across N hosts. Hyperspace
  is a strong distributed-agent infrastructure but **not** a
  single-model sharding system; llama.cpp `--rpc` is a
  defensible alternative but with smaller Apple-Silicon
  optimisation.
- **Realistic model class on 2×36 GB:** 70B-class in Q4
  (≈ 40 GB weights; fits across two Macs with KV-cache headroom).
  35B-class in Q4 fits on a single Mac; sharding buys
  context-length / KV headroom only.
- **Wevra integration boundary** (`vision_mvp.wevra.llm_backend`):
  a duck-typed `LLMBackend` Protocol with two concrete
  implementations (`OllamaBackend`, `MLXDistributedBackend`).
  The runtime's inner-loop seal-PROMPT / seal-LLM_RESPONSE chain
  accepts any conformant backend without any spine modification
  (W5-2 proved); the OpenAI-compatible wire shape is locked
  against a stub server (W5-3 proved).
- **Real cross-LLM parser-boundary measurement (W5-1)**:
  `parser_boundary_real_llm.py` against the live Mac 1 Ollama
  endpoint yields cross-model PARSE_OUTCOME failure-kind
  TVD = 1.000 between Qwen-2.5-14B (dense, Q4) and Qwen-3.5-35B
  (MoE, Q4, `think=False`) under strict parsing on n=10
  instances — the larger model emits OLD/NEW close as `<<`
  instead of `<<<` and lands entirely in `unclosed_new`, while
  the smaller model emits `<<<` cleanly. Robust-mode
  `recovery=closed_at_eos` collapses cross-model TVD to 0.000.
  This **inverts the naive prediction** that a stronger model
  would reduce parser-boundary instability.

### Active conjectures (SDK v3.6)

- **W5-C1**: parser-boundary instability is a (model
  architecture × prompt-format) interaction, not a capacity
  artefact. Empirical-research; falsifier = a bank on which
  the larger model strict-parses ok > 50%.
- **W5-C2**: robust-mode `recovery=closed_at_eos` is the
  load-bearing safety net that makes the capsule-native runtime
  model-class-agnostic on the bundled prompt format. Empirical-
  research; falsifier = a model whose `unclosed_new` cannot be
  salvaged.
- **W5-C3**: closed-vocabulary `PARSE_OUTCOME.failure_kind` is
  a *minimum sufficient* typed witness of cross-model behaviour
  differences. Conjectural research; falsifier = a model pair
  with identical strict-mode `failure_kind` distribution but
  materially different downstream test-pass rate.

## Current frontier (SDK v3.5, 2026-04-26)

### Active moves (SDK v3.5 — multi-agent capsule coordination)

- **Capsule-native multi-agent team coordination
  (W4 family).** Three new closed-vocabulary capsule kinds
  (TEAM_HANDOFF, ROLE_VIEW, TEAM_DECISION) make capsules
  load-bearing *between* agents. ``TeamCoordinator`` drives one
  coordination round end-to-end; ``audit_team_lifecycle``
  mechanically verifies T-1..T-7 (Theorem W4-1).
- **Coverage-implies-correctness** (W4-2, proved-conditional) and
  **Local-view limitation** (W4-3, proved-negative) anchor the
  team-level mechanism in the formal layer.
- **Learned per-role admission policy** (``team_policy.py``)
  strictly improves pooled team-decision accuracy over the
  strongest fixed baseline at matched per-role budgets on the
  Phase-52 incident-triage bench (W4-C1 positive empirical;
  conjectural at smaller train scales).
- **Phase-52 reference benchmark**
  (``vision_mvp/experiments/phase52_team_coord.py``) compares
  substrate / capsule_fifo / capsule_priority / capsule_coverage
  / capsule_learned head-to-head and reports
  ``audit_ok_rate = 1.000`` for every capsule strategy.

### Active moves (SDK v3.4 — still in force)

- **Capsule-native execution one further structural layer past
  v3.3.** PROMPT capsule sealed for every LLM call's prompt
  bytes; LLM_RESPONSE capsule sealed for every response bytes.
  PROMPT.parents = (SWEEP_SPEC,) (Theorem W3-42); LLM_RESPONSE
  parent = sealed PROMPT (Theorem W3-43); PARSE_OUTCOME may
  parent on (SWEEP_SPEC, LLM_RESPONSE) so the
  prompt → response → parse → patch → verdict chain is a
  typed DAG witness end-to-end (Theorem W3-44).
- **Lifecycle audit extended to L-9 / L-10 / L-11** (Theorem
  W3-45). Soundness: ``audit_capsule_lifecycle(ctx).verdict ==
  "OK"`` iff the ledger satisfies the eleven invariants.
- **Synthetic-LLM mode for CI-runnable end-to-end exercise.**
  ``SweepSpec(mode="synthetic", synthetic_model_tag=<tag>)``
  uses a deterministic in-process synthetic LLM client; no
  network. The full prompt/response/parse/patch/verdict chain
  seals end-to-end on every (task, strategy).
- **Cross-model parser-boundary research (W3-C6, empirical).**
  ``vision_mvp.experiments.parser_boundary_cross_model``
  reports cross-distribution PARSE_OUTCOME failure-kind TVD up
  to 1.000 across the synthetic distribution library, and
  strict→robust parser-mode shift up to 1.000 on
  ``synthetic.unclosed``.

### Active moves (SDK v3.3 — still in force)

- **PARSE_OUTCOME lifecycle gate.** Theorem W3-39.
- **Runtime-checkable lifecycle audit.** Theorem W3-40 / W3-45.
- **Deterministic-mode replay.** Theorem W3-41.

### Sharp limitation theorems we hold

- **W3-14** (negative): per-capsule budgets cannot enforce
  table-level cardinality invariants.
- **W3-16** (negative): cohort-lifting cannot enforce relational
  invariants.
- **W3-17** (conditional): admission rules cannot exceed the
  priority-decoder ceiling under ceiling-forcing spurious
  injection.
- **W3-21** (negative): linear class-agnostic decoders cannot
  achieve symmetric zero-shot transfer when gold-conditional
  feature signs flip across domains.
- **W3-29** (lower bound): magnitude-monoid linear family cannot
  cross the Bayes-divergence zero-shot risk lower bound.
- **W3-36** (sharp impossibility): the primary capsule ledger
  cannot authenticate its own rendering's bytes.
- **W4-3 (SDK v3.5)** (proved-negative): per-role budget below
  the role's causal-share floor admits sound runs that fail the
  team gate; no admission policy (FIFO, priority, coverage,
  learned) can recover. The natural next move is a
  cohort-lifted role view (W4-C2, conjectural).

### Active conjectures

- **W3-C1**: every Phase-N bounded-context theorem subsumes under
  the capsule contract. Conjectural (the four-case subsumption is
  proved; the general statement is open).
- **W3-C5 (legacy SDK v3.3)**: a sub-intra-cell PROMPT /
  LLM_RESPONSE capsule slice closes the inner-loop boundary
  without breaking W3-34 spine equivalence. **DISCHARGED in
  SDK v3.4** by Theorems W3-42 / W3-43 / W3-44 / W3-45.
- **W3-C6 (new in SDK v3.4)**: synthetic-LLM cross-distribution
  PARSE_OUTCOME failure-kind TVD ≥ 0.5 across the calibrated
  distribution library. **Empirical** (reproducible; not a
  proof — the distribution library is synthetic, not real
  cross-LLM).
- **W3-C9**: refined paradigm-shift reading (Phase-49 candidate at
  $n=80$ point-estimate, zero-shot gap reading).
- **W3-C10**: relational decoder level-ceiling.
- **W4-C1 (SDK v3.5)**: learned per-role admission policy
  admits strictly fewer handoffs (12/12 seeds, robust direction)
  and improves pooled team-decision accuracy on most train seeds
  (gap_full > 0 in 11/12 seeds, mean +0.054; gap_root_cause
  > 0 in 8/12 seeds, mean +0.032) over the strongest fixed
  admission baseline (coverage-guided) on the Phase-52 default
  config — but the accuracy advantage reverses at higher noise
  (spurious=0.50). Empirical: budget-efficiency dominance is
  robust per-seed; accuracy advantage is mean-positive, not
  strict per-seed.
- **W4-C2 (SDK v3.5)**: cohort-lifted role view closes W4-3 on a
  sub-class of scenarios.
- **W4-C3 (SDK v3.5)**: capsule-layer admission rule subsumes
  the Phase-36 ``AdaptiveSubscriptionTable`` route-edit primitive.

### Active retractions

- **W3-C7 (strict reading) is retracted.** "Point-estimate
  Gate 1 at $\hat p \ge 0.400$ AND strict zero-shot Gate 2 with
  per-direction penalty ≤ 5pp" was falsified at $n=320$ (W3-26,
  W3-27). Do not reintroduce the strict bar.
- **W3-C3** is retracted in favour of W3-15 cohort lift.
- **The earlier W3-C4** (now reused for SDK-v3.3
  PARSE_OUTCOME conjecture) named a candidate decoder paradigm
  shift; the strict reading is folded into W3-C7 retraction.

## What we are NOT actively claiming

- **Not** "we solved context."
- **Not** "we solved multi-agent context." SDK v3.5 makes the
  capsule abstraction load-bearing at the **team boundary** on
  one synthetic benchmark family (Phase-52 incident-triage) under
  a deterministic team decoder. The W4-2 result is
  proved-conditional (premises: faithful decoder + sound
  admission); the W4-C1 learned-policy advantage is empirical-
  positive on the default config and conjectural on smaller
  training scales. External validity to real production
  multi-agent teams is open.
- **Not** "the runtime is fully capsule-native." Specifically not
  capsule-native: sandbox stdout/stderr, sub-step parser-internal
  objects (regex match objects, recovery heuristic intermediate
  state), and on-the-wire LLM streaming chunks. PROMPT bytes and
  LLM_RESPONSE bytes ARE now capsule-tracked under SDK v3.4 (the
  prior "not capsule-native: LLM prompt bytes, raw LLM response
  bytes" line is **superseded** by Theorems W3-42 / W3-43).
- **Not** "Wevra is a universal multi-agent platform."
- **Not** "the decoder beat the Phase-31 ceiling by 22.5 pp."
  The sharper reading is W3-19 ($+15$pp at $n=80$, FIFO admission).
- **Not** "deterministic mode means the entire run is
  reproducible." It means the *capsule DAG* is reproducible under
  a frozen JSONL + a deterministic profile. Wall-clock and
  host-local fields are stripped from CIDs but live on disk.
- **Not** "the synthetic-LLM cross-distribution study is a real
  cross-LLM study." The distributions are calibrated synthetic
  (see ``synthetic_llm.SYNTHETIC_MODEL_PROFILES``), not
  measurements of ``gemma2:9b`` / ``qwen2.5:7b`` outputs. The
  empirical claim is about the parser's failure-kind closed
  vocabulary's *resolving power*, not about LLM output
  distributions in the wild.

## How to update this document

1. When a phase ships, add one line to the "Active moves" list and
   move any superseded line to "Sharp limitation theorems we hold"
   or "Active retractions" as appropriate.
2. When a conjecture is proved or falsified, move it to the
   correct section and update `THEOREM_REGISTRY.md`.
3. When a milestone note ships, add a one-line cross-link in this
   doc's relevant section.
4. Always update the "Last touched" date at the top.

## Cross-references

- Formal model (run-boundary, W3 family): `docs/CAPSULE_FORMALISM.md`
- Formal model (team-boundary, W4 family): `docs/CAPSULE_TEAM_FORMALISM.md`
- Theorem registry: `docs/THEOREM_REGISTRY.md`
- How-not-to-overstate rules: `docs/HOW_NOT_TO_OVERSTATE.md`
- Master plan: `docs/context_zero_master_plan.md`
- Milestone notes: `docs/archive/wevra-milestones/RESULTS_WEVRA_*.md`,
  `docs/archive/capsule-research/RESULTS_CAPSULE_*.md` (historical),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_DEEP_INTRA_CELL.md` (SDK v3.3),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_INNER_LOOP.md` (SDK v3.4),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_TEAM_COORD.md` (SDK v3.5),
  `docs/archive/wevra-milestones/RESULTS_WEVRA_DISTRIBUTED.md` (SDK v3.6),
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` (SDK v3.7 — this milestone)
- Paper draft: `papers/wevra_capsule_native_runtime.md`
- Tests: `vision_mvp/tests/test_wevra_capsule_native*.py`,
  `test_wevra_capsule_native_deeper.py`,
  `test_wevra_capsule_native_inner_loop.py` (SDK v3.4),
  `test_wevra_team_coord.py` (SDK v3.5 — multi-agent slice),
  `test_capsule_*.py`
- Cross-model parser-boundary experiment:
  `vision_mvp/experiments/parser_boundary_cross_model.py`
- Multi-agent team coordination benchmark (synthetic):
  `vision_mvp/experiments/phase52_team_coord.py`
- Stronger-model multi-agent benchmark (real LLM):
  `vision_mvp/experiments/phase53_scale_vs_structure.py`
- MLX distributed runbook (operator path for Mac 2):
  `docs/MLX_DISTRIBUTED_RUNBOOK.md`
