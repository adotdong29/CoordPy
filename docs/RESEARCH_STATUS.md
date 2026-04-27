# Research status — canonical, current

> Single-source-of-truth for the *active* research position of the
> Context Zero programme. If this file disagrees with any other
> doc on what is *true now*, this file is right and the other file
> is stale. For *theorem-by-theorem* status, see
> `docs/THEOREM_REGISTRY.md`. For *what may be claimed*, see
> `docs/HOW_NOT_TO_OVERSTATE.md`. Last touched: SDK v3.12,
> 2026-04-26.

## TL;DR

The programme now has **eight** coupled research axes, each with a
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
7. **Cross-role cohort-coherence multi-agent coordination**
   — *active, new (SDK v3.8)*. **Phase-54** benchmark
   (`vision_mvp/experiments/phase54_cross_role_coherence.py`)
   directly attacks the SDK v3.7 Phase-53 failure mode by
   redesigning the regime so structure has a real chance: a
   deterministic candidate stream where each scenario has one
   ``real_service`` (gold) and one ``decoy_service`` (foreign);
   each producer role emits ``service=<tag>``-tagged candidates
   with the gold tag in **strict plurality**; the auditor sees
   surplus candidates above ``K_auditor=4`` (``5 ≤ |candidates| ≤ 7``).
   The new admission policy
   ``CohortCoherenceAdmissionPolicy`` (in
   ``vision_mvp.wevra.team_coord``) provides two sub-modes:
   *streaming* (running cohort over admitted) and *buffered*
   (pre-fitted plurality from candidate stream's payloads via
   ``from_candidate_payloads``). Headline: at the pre-committed
   default, ``capsule_cohort_buffered`` achieves
   ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   and ``capsule_cohort_streaming`` all produce 0.000 — a
   **+1.000** structural win, stable across 5/5 alternate bank
   seeds. The W7 family (W7-1 / W7-1-aux / W7-2 /
   W7-2-conditional / W7-3 — proved or proved-empirical) anchors
   the milestone formally; the W7-C family makes the multi-service
   / decoder-side / real-LLM extensions falsifiable. **Honest
   scope:** the structural win is *conditional* on the bench
   property (gold-plurality + foreign-service decoys + budget
   surplus); the streaming variant is unstable and ties FIFO
   (W7-1-aux); W7-3 is the extraction floor — admission cannot
   recover claims the producer never emitted (the Phase-53
   ``deadlock_pool_exhaustion`` failure case).

9. **Multi-service top-K cross-role corroboration multi-agent
   coordination** — *active, new (SDK v3.10)*. **Phase-56**
   benchmark
   (`vision_mvp/experiments/phase56_multi_service_corroboration.py`)
   directly attacks the W8 *multi-service-gold* falsifier by
   building the smallest deterministic regime where (i) every
   scenario has ``gold_services`` of size 2 (multi-service incident),
   (ii) both gold services are corroborated by ≥ 2 distinct producer
   roles each, AND (iii) a decoy service has raw plurality but is
   corroborated by exactly 1 producer role. 10/10 default scenarios
   satisfy all three properties; mechanically verified by
   ``Phase56BankShapeTests``. The new admission policy
   ``MultiServiceCorroborationAdmissionPolicy`` admits the **top-K
   cross-role-corroborated tier** (default ``top_k=2,
   min_corroborated_roles=2``) via the argmax-by-role-count gate —
   strictly generalising W8 single-tag corroboration. Headline: at
   the pre-committed default (``K_auditor=4``, ``T_auditor=128``,
   ``n_eval=10``, ``bank_seed=11``), ``capsule_multi_service``
   achieves ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   ``capsule_cohort_buffered`` (W7-2), AND
   ``capsule_corroboration`` (W8) all produce 0.000 — the **first
   strict separation between multi-service top-K corroboration and
   single-tag corroboration**, **+1.000** vs the SDK v3.9 strongest
   method, stable across **5/5** alternate bank seeds. The W9
   family (W9-1 / W9-2 / W9-3 / W9-4 — proved or proved-empirical)
   anchors the milestone formally; the W9-C family makes the
   bundle-aware decoder / |gold|≥3 / real-LLM extensions
   falsifiable. **Honest scope:** the win is *conditional* on the
   named bench property (multi-service-gold + single-role-decoy);
   W9-4 is the named falsifier regime where the decoy is also
   corroborated and W9 ties FIFO at 0.000; W9-3 backward-compat
   preserves W8 on Phase 55 and W7-2 on Phase 54; no regression on
   Phase 53 synthetic. The milestone clears the **strong success
   bar** of `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1
   (R-56 anchor) — a strict gain ≥ 0.20 on Phase 56 vs both
   substrate FIFO and SDK v3.9 W8, stable across ≥ 3 seeds, no
   regression on R-53 / R-54 / R-55, audit T-1..T-7 preserved on
   every cell, named bench property + named falsifier regime.

8. **Cross-role corroboration multi-agent coordination**
   — *active, new (SDK v3.9)*. **Phase-55** benchmark
   (`vision_mvp/experiments/phase55_decoy_plurality.py`) directly
   attacks the W7-2 falsifier by building the smallest deterministic
   regime where (i) some decoy service has *strictly more raw
   mentions* than gold (so W7-2 single-tag plurality picks the
   decoy and ties FIFO at 0.000) AND (ii) the gold service is
   *cross-role corroborated* — mentioned by strictly more distinct
   producer roles than any decoy. 10/10 default scenarios satisfy
   both properties; mechanically verified by
   ``Phase55BankShapeTests``. The new admission policy
   ``CrossRoleCorroborationAdmissionPolicy`` aggregates over
   (role, tag) bipartite multisets via the score function
   ``score(tag) = role_weight·|distinct_roles(tag)| +
   |raw_mentions(tag)|``. Headline: at the pre-committed default
   (``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
   ``bank_seed=11``), ``capsule_corroboration`` achieves
   ``accuracy_full = 1.000`` while substrate FIFO,
   ``capsule_fifo``, ``capsule_priority``, ``capsule_coverage``,
   AND ``capsule_cohort_buffered`` (W7-2) all produce 0.000 —
   the **first strict separation** between cross-role corroboration
   and W7-2 single-tag plurality, **+1.000** vs both baselines,
   stable across **5/5** alternate bank seeds. The W8 family
   (W8-1 / W8-2 / W8-3 / W8-4 — proved or proved-empirical)
   anchors the milestone formally; the W8-C family makes the
   multi-service / decoder-side / real-LLM extensions falsifiable.
   **Honest scope:** the win is *conditional* on the named bench
   property (decoy-plurality + cross-role-corroborated gold);
   W8-4 is the named falsifier regime where the decoy is
   corroborated and corroboration ties FIFO at 0.000;
   W8-3 backward-compat preserves W7-2 on Phase 54 (corroboration
   matches cohort_buffered at 1.000); no regression on Phase 53
   synthetic (0.800) or Phase 53 14B real-LLM (0.800). The
   milestone clears the **strong success bar** of
   `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` § 1.1 — a
   strict gain ≥ 0.20 on Phase 55, stable across ≥ 3 seeds, no
   regression on Phase 53 / Phase 54, audit T-1..T-7 preserved
   on every cell, named bench property + named falsifier regime.

6. **Model-scale vs capsule-structure on multi-agent
   coordination** — *active, settled (SDK v3.7)*. **Phase-53**
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

## Current frontier (SDK v3.12, 2026-04-26)

### Active moves (SDK v3.12 — multi-round bundle-aware team decoder + W11 family — *first cross-round coordination move*)

- **Phase-58 multi-round delayed-causal-evidence benchmark.**
  ``vision_mvp.experiments.phase58_multi_round_decoder`` runs 9
  strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9),
  capsule_bundle_decoder (W10 single-round), capsule_multi_round
  (W11)) on a deterministic 8-scenario bank with the
  **delayed-causal-evidence** property: round-1 carries
  generic-noise-only mentions of (gold_A, gold_B, decoy) where the
  decoy is cross-role-corroborated; round-2 carries one
  specific-tier disambiguating ``claim_kind`` with NO ``service=``
  token. 8/8 default scenarios satisfy the property; mechanically
  verified by ``Phase58BankShapeTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=8``,
  ``bank_seed=11``, ``bank_replicates=2``,
  ``noise_decoy_role_floor=2``. Headline:
  ``capsule_multi_round`` achieves ``accuracy_full = 1.000`` while
  substrate, FIFO, priority, coverage, W7-2, W8, W9, AND W10
  single-round all produce ``accuracy_full = 0.000`` — the **first
  strict separation between multi-round and single-round capsule-
  native coordination** in the programme, **+1.000** vs every
  prior method, stable across **5/5** alternate bank seeds. The
  W11 family (W11-Λ / W11-1 / W11-2 / W11-3 / W11-4 — proved or
  proved-empirical) anchors the milestone formally; the W11-C
  family (W11-C1/C2/C3) makes the cross-bench / real-LLM /
  multi-step extensions falsifiable.
- **``MultiRoundBundleDecoder`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Decodes the *union* of
  admitted handoffs across multiple ROLE_VIEW capsules; if the
  elected root_cause is specific-tier, drops every service tag
  whose admitted mentions are exclusively generic-noise kinds AND
  span ≥ ``noise_decoy_role_floor`` distinct producer roles
  (default 2). Inner ``BundleAwareTeamDecoder`` configured with
  ``cck_filter=False`` so the contradiction-aware step is the only
  filter; the W10 fallback path preserves single-round wins on
  R-54..R-57 (W11-3). Companion helper
  ``collect_admitted_handoffs(ledger, role_view_cids)`` materialises
  multi-round admitted handoffs into the duck-typed
  ``_DecodedHandoff`` shape. Re-exported as
  ``MultiRoundBundleDecoder`` and ``collect_admitted_handoffs``.
- **Theorem family W11.** W11-Λ (single-round structural limit on
  R-58, proved-empirical + structural sketch), W11-1 (multi-round
  decoder sufficiency, proved-empirical n=40 saturated across 5
  seeds), W11-2 (round-union monotonicity, proved structural),
  W11-3 (backward-compat with W7-2 / W8 / W9 / W10 on
  R-54 / R-55 / R-56 / R-57, proved-empirical), W11-4 (round-budget
  falsifier, proved-empirical n=8 saturated). The W11-C family
  (W11-C1/C2/C3) makes the cross-bench / real-LLM / multi-step
  extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-58 anchor +
  bar 8 — temporal/structural split). The SDK v3.12 result clears
  the **strong success bar** § 1.1 (strict gain ≥ 0.20 on R-58 vs
  every SDK v3.11 single-round method, stable across ≥ 3 seeds, no
  regression on R-53 / R-54 / R-55 / R-56 / R-57, audit T-1..T-7
  preserved on every cell, named bench property + named falsifier
  regime, AND temporal/structural split bar 8 satisfied).
- **Honest scope.** The W11-1 win is *conditional* on the named
  bench property; the W11-4 falsifier regime is the explicit
  counterexample. W11-3 backward-compat is exact on R-54 / R-55 /
  R-56 / R-57 thanks to (a) the inner W10 decoder's fallback-on-
  small-admitted-set path, (b) the noise-decoy floor being
  insensitive to single-role decoys. The contradiction-aware drop
  is closed-vocabulary on incident-triage; W11-C1 is the
  conjectural extension to other benchmark families. The decoder
  is a no-op on generic-tier elected root_cause (W11-Λ at the
  temporal axis collapses).

### Active conjectures (SDK v3.12)

- **W11-C1**: noise-decoy drop generalises to non-incident-triage
  benchmark families. Conjectural; falsifier = a benchmark family
  where a generic-noise-only mention is informative.
- **W11-C2**: real-LLM transfer of W11-1. Conjectural; Phase-59
  candidate.
- **W11-C3**: contradiction-aware round-resolution rule (last-wins
  / weighted-confidence) strictly outperforms naive union with
  ≥ 3 rounds and conflicting specific-tier evidence across rounds.
  Conjectural; multi-step capsule chains not yet shipped.

### Discharged conjectures (SDK v3.12)

- **W10-C3** (SDK v3.11): multi-round bundle decoder closes W10-4
  on a sub-class of scenarios. **PARTIALLY DISCHARGED** by the
  W11 family on R-58 (a different regime than W10-4): W11-1
  resolves the *delayed-causal-evidence* sub-class; the W10-4
  *decoy-CCK-promotion* sub-class remains open (the W11
  contradiction-aware drop is orthogonal — drops noise-corroborated
  decoys, not CCK-corroborated decoys).

## Previous frontier (SDK v3.11, 2026-04-26)

### Active moves (SDK v3.11 — bundle-aware team decoder + W10 family — *first decoder-side coordination move*)

- **Phase-57 multi-service-gold + corroborated-decoy decoder-forcing
  benchmark.**
  ``vision_mvp.experiments.phase57_decoder_forcing`` runs 8 strategies
  (substrate, capsule_fifo, capsule_priority, capsule_coverage,
  capsule_cohort_buffered (W7-2), capsule_corroboration (W8),
  capsule_multi_service (W9), capsule_bundle_decoder (W10-AD: W9
  admission + bundle-aware decoder)) on a deterministic 12-scenario
  bank (clipped to 10 by default) where (i) every scenario has
  ``gold_services`` of size 2 with a *specific-tier* gold root_cause,
  (ii) both gold services are corroborated by ≥ 2 distinct producer
  roles via at least one CCK-eligible (causal-tier) claim_kind, AND
  (iii) a decoy service is *also* corroborated by ≥ 2 distinct
  producer roles via *only* non-causal claim_kinds (LATENCY_SPIKE,
  ERROR_RATE_SPIKE, FW_BLOCK_SURGE). 10/10 default scenarios satisfy
  the bench property; mechanically verified by
  ``Phase57BankShapeTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=10``,
  ``bank_seed=11``, ``ms_top_k=3``, ``ms_min_corroborated_roles=2``,
  ``bundle_role_floor=1``,
  ``fallback_admitted_size_threshold=2``. Headline:
  ``capsule_bundle_decoder`` achieves ``accuracy_full = 1.000``
  while substrate, FIFO, priority, coverage, W7-2, W8, AND W9 all
  produce ``accuracy_full = 0.000`` — the **first strict separation
  between decoder-side coordination and any service-blind admission
  policy** in the programme, **+1.000** vs every admission baseline,
  stable across **5/5** alternate bank seeds. The W10 family
  (W10-Λ / W10-1 / W10-2 / W10-3 / W10-4 — proved or proved-empirical)
  anchors the milestone formally; the W10-C family (W10-C1/C2/C3)
  makes the cross-bench / real-LLM / multi-round extensions
  falsifiable.
- **``BundleAwareTeamDecoder`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Three knobs (``cck_filter``,
  ``role_corroboration_floor``,
  ``fallback_admitted_size_threshold``). Closed-vocabulary CCK table
  ``CAUSAL_CLAIM_KINDS_PER_ROOT_CAUSE`` maps each root_cause label
  to its causal claim-kind set (data-tier / storage-tier / compute /
  edge / network / generic). Decoder rule: pick root_cause via the
  same priority decoder as
  ``incident_triage._decoder_from_handoffs``; project admitted
  service tags through the CCK predicate; trust admission when the
  admitted-tag set is small (size ≤ threshold). Deterministic,
  training-free, additive on top of the existing ``TeamCoordinator``
  / ``ROLE_VIEW`` / ``TEAM_DECISION`` capsule layer. Re-exported as
  ``BundleAwareTeamDecoder`` and ``decode_admitted_role_view``.
- **Theorem family W10.** W10-Λ (admission-only structural limit on
  R-57, proved-empirical + structural sketch), W10-1 (bundle-decoder
  sufficiency, proved-empirical n=50 saturated), W10-2 (CCK
  structural correctness, proved by inspection), W10-3 (backward-
  compat with W7-2 / W8 / W9 on R-54 / R-55 / R-56, proved-empirical),
  W10-4 (decoy-CCK-promotion falsifier, proved-empirical n=10
  saturated). The W10-C family makes the cross-bench / real-LLM /
  multi-round extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-57 anchor +
  bar 7 — admission/decoding split). The SDK v3.11 result clears
  the **strong success bar** § 1.1 (strict gain ≥ 0.20 on R-57 vs
  both substrate FIFO and SDK v3.10 W9, stable across ≥ 3 seeds, no
  regression on R-53 / R-54 / R-55 / R-56, audit T-1..T-7 preserved
  on every cell, named bench property + named falsifier regime,
  AND admission/decoding split bar 7 satisfied).
- **Honest scope.** The W10-1 win is *conditional* on the named
  bench property; the W10-4 falsifier regime is the explicit
  counterexample. W10-3 backward-compat is exact on R-54 / R-55 /
  R-56 thanks to the trust-admission fallback (size ≤ 2 threshold).
  The CCK table is *closed-vocabulary for incident-triage*; W10-C1
  is the conjectural extension to other benchmark families. The
  bundle decoder is a no-op on generic-tier root_causes
  (``error_spike`` / ``latency_spike``) — this is the named scope
  limit of W10-1.

### Active conjectures (SDK v3.11)

- **W10-C1**: CCK table extends to non-incident-triage benchmark
  families (security incident, robotics, compliance review).
  Conjectural; falsifier = a benchmark family where no closed-
  vocabulary tier mapping exists.
- **W10-C2**: real-LLM transfer of W10-1. Conjectural; Phase-58
  candidate.
- **W10-C3**: multi-round bundle decoder closes W10-4 on a
  sub-class of scenarios. Conjectural; multi-round capsule chain
  not yet shipped.

### Discharged conjectures (SDK v3.11)

- **W9-C1** (SDK v3.10): bundle-aware decoder companion strictly
  improves on Phase-56 falsifier. **DISCHARGED-empirical** by W10-1
  on Phase 57 (+1.000 vs every admission-only baseline). The
  decoder-side axis is now the load-bearing axis of the SDK v3.11
  milestone.

## Previous frontier (SDK v3.10, 2026-04-26)

### Active moves (SDK v3.10 — multi-service top-K cross-role corroboration multi-agent benchmark + W9 family)

- **Phase-56 multi-service-gold + cross-role-corroborated benchmark.**
  ``vision_mvp.experiments.phase56_multi_service_corroboration`` runs
  7 admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9)) on a
  deterministic 10-scenario bank with the **multi-service-gold +
  both-gold-cross-role-corroborated + single-role-decoy-storm**
  properties (10/10 scenarios). Pre-committed default:
  ``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
  ``bank_seed=11``, ``ms_top_k=2``, ``ms_min_corroborated_roles=2``.
  Headline: ``multi_service − fifo accuracy_full = +1.000``,
  ``multi_service − cohort_buffered = +1.000``, AND
  ``multi_service − corroboration = +1.000``, stable across 5/5
  alternate bank seeds. The **first strict separation between
  multi-service top-K corroboration and single-tag corroboration**
  in the programme.
- **``MultiServiceCorroborationAdmissionPolicy`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Two sub-modes (streaming /
  buffered). Buffered factory ``from_candidate_stream`` is the W9-1
  anchor — pre-fits a top-K dominant tag set via the
  argmax-by-role-count tier of the corroboration score function.
  Selection rule: ``min_corroborated_roles`` floor → argmax-by-role-
  count tier → top-K by score (lex tie-break). Deterministic,
  training-free, one regex + two counters + the ``_dominant_tag_set``
  helper. Re-exported as
  ``TeamMultiServiceCorroborationAdmissionPolicy``.
- **Theorem family W9.** W9-1 (strict separation, proved-empirical
  n=50 saturated), W9-2 (argmax-tier strict-ordering, proved
  structural), W9-3 (backward-compat with W8 + W7-2 on Phase 55 +
  Phase 54, proved-empirical), W9-4 (decoy-corroboration falsifier,
  proved-empirical n=10 saturated). The W9-C family (W9-C1/C2/C3)
  makes the bundle-aware decoder / |gold|≥3 / real-LLM extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-56 anchor).
  The SDK v3.10 result clears the **strong success bar** § 1.1
  (strict gain ≥ 0.20 on R-56 vs both substrate FIFO and SDK v3.9
  W8, stable across ≥ 3 seeds, no regression on R-53 / R-54 / R-55,
  audit T-1..T-7 preserved on every cell, named bench property +
  named falsifier regime).
- **Honest scope.** The W9-1 win is *conditional* on the named
  bench property; the W9-4 falsifier regime is the explicit
  counterexample. W9-3 preserves the SDK v3.9 W8-1 win
  byte-for-byte on Phase 55 (via the argmax-by-role-count gate).

### Active conjectures (SDK v3.10)

- **W9-C1** (new SDK v3.10): bundle-aware decoder companion that
  filters service tags at decode time by the dominant
  *(claim_kind, role)* signature strictly improves accuracy_full on
  the Phase-56 falsifier regime. **Conjectural**; reframes W8-C3 as
  the natural attack on the W9-4 falsifier — pushes the structural
  axis from admission to decoding.
- **W9-C2** (new SDK v3.10): top-K extension to ``|gold| ≥ 3``.
  Conjectural; Phase-57 candidate; the policy already supports
  arbitrary ``top_k``.
- **W9-C3** (new SDK v3.10): real-LLM transfer of W9-1.
  Conjectural; SDK v3.10 confirms no-regression in low-surplus
  synthetic regime.

### Discharged conjectures (SDK v3.10)

- **W8-C1** (SDK v3.9): top-k corroboration improves multi-service
  scenarios. **DISCHARGED-empirical** by W9-1 on Phase 56 (+1.000).

## Previous frontier (SDK v3.9, 2026-04-26)

### Active moves (SDK v3.9 — cross-role corroboration multi-agent benchmark + W8 family)

- **Phase-55 decoy-plurality + cross-role-corroborated benchmark.**
  ``vision_mvp.experiments.phase55_decoy_plurality`` runs 6
  admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8)) on a deterministic 10-scenario bank
  with the **decoy-plurality + gold-corroboration** properties.
  Pre-committed default: ``K_auditor=4``, ``T_auditor=128``,
  ``n_eval=10``, ``bank_seed=11``. Headline: ``corroboration −
  fifo accuracy_full = +1.000`` AND ``corroboration −
  cohort_buffered accuracy_full = +1.000``, stable across 5/5
  alternate bank seeds. The first strict separation between W8
  and W7-2 in the programme.
- **``CrossRoleCorroborationAdmissionPolicy`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Two sub-modes (streaming /
  buffered). Buffered factory ``from_candidate_stream`` is the
  W8-1 anchor — pre-fits a (role, tag)-aggregated dominant tag
  via score function ``W_role · |distinct_roles| + |raw_mentions|``.
  Deterministic, training-free, one regex + two counters.
  Re-exported as ``TeamCrossRoleCorroborationAdmissionPolicy``.
- **Theorem family W8.** W8-1 (strict separation, proved-empirical
  n=50 saturated), W8-2 (score-function strict-ordering, proved
  structural), W8-3 (backward-compat with W7-2 on Phase 54,
  proved-empirical), W8-4 (decoy-corroboration falsifier,
  proved-empirical n=10 saturated). The W8-C family
  (W8-C1/C2/C3) makes the multi-service / decoder-side / real-LLM
  extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``. The SDK v3.9
  result clears the **strong success bar** § 1.1 (strict gain
  ≥ 0.20 on R-55, stable across ≥ 3 seeds, no regression on R-53
  / R-54, audit preserved on every cell, named bench property +
  named falsifier regime).
- **Honest scope.** The W8-1 win is *conditional* on the named
  bench property; the W8-4 falsifier regime is the explicit
  counterexample. W8-3 preserves the SDK v3.8 W7-2 win
  byte-for-byte on Phase 54.

### Active conjectures (SDK v3.9)

- **W8-C1**: multi-service-gold extension (top-k corroboration).
  conjectural; falsifier in Phase-56 candidate.
- **W8-C2**: W8-1 transfers to a real-LLM regime when the
  producer LLM emits multi-service decoy chatter that satisfies
  the bench property. conjectural; Phase-56 candidate.
- **W8-C3**: bundle-aware decoder + corroboration admission
  strictly outperforms admission alone on partial-coverage
  benches. conjectural; restates W7-C2 under W8 framing.

## Previous frontier (SDK v3.8, 2026-04-26)

### Active moves (SDK v3.8 — cross-role cohort-coherence multi-agent benchmark + W7 family)

- **Phase-54 cross-role cohort-coherence benchmark.**
  ``vision_mvp.experiments.phase54_cross_role_coherence`` runs 5
  admission strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_streaming, capsule_cohort_buffered)
  on a deterministic 10-scenario bank with the **gold-plurality**
  property (gold service has strictly more auditor-routed
  candidates than any decoy service). Pre-committed default:
  ``K_auditor=4``, ``T_auditor=128``, ``n_eval=10``,
  ``bank_seed=11``. Headline: ``cohort_buffered − fifo
  accuracy_full = +1.000``, stable across 5/5 alternate bank
  seeds.
- **``CohortCoherenceAdmissionPolicy`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Two sub-modes:
  *streaming* (arrival-order-sensitive baseline) and *buffered*
  (arrival-order-stable, pre-fitted plurality via
  ``from_candidate_payloads``). Deterministic, training-free,
  one regex + one counter. Re-exported as
  ``TeamCohortCoherenceAdmissionPolicy``.
- **Theorem family W7.** W7-1 (FIFO unbeatability under low
  surplus, proved-empirical anchor on Phase-53), W7-1-aux
  (streaming cohort instability under arrival permutation,
  proved-empirical), W7-2 (cohort buffered structural win under
  gold-plurality, proved-empirical n=50 saturated), W7-2-conditional
  (K-sweep window, proved-empirical), W7-3 (extraction floor,
  proved-negative — corollary of Capsule Contract C5). The
  W7-C family (W7-C1/C2/C3) makes the multi-service /
  decoder-side / real-LLM extensions falsifiable.
- **Honest scope.** The W7-2 win is *conditional* on stated bench
  properties (gold-plurality + cross-role coherence +
  ``|candidates| > K_auditor``); it does not generalise to every
  multi-agent benchmark. The capsule layer's *audit* contribution
  (T-1..T-7) is preserved and extends to Phase-54 unchanged.

### Active conjectures (SDK v3.8)

- **W7-C1**: multi-service-gold extension (top-2 plurality).
  conjectural; falsifier in Phase-55 candidate.
- **W7-C2**: bundle-aware decoder + cohort admission strictly
  dominates cohort admission alone on weak-coherence benches.
  conjectural; not yet measured.
- **W7-C3**: W7-2 transfers to the real-LLM regime when the LLM
  is prompted with a multi-service event mix. conjectural;
  Phase-56 candidate.

## Previous frontier (SDK v3.7, 2026-04-26)

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
- **Not** "we solved multi-agent context." SDK v3.12's W11-1 result
  is the strongest cross-regime structural-win the programme has
  produced (multi-round bundle decoder wins on R-58 by +1.000 vs
  every single-round method including SDK v3.11 W10;
  backward-compatible on R-54 / R-55 / R-56 / R-57; no regression
  on R-53; stable across 5/5 bank seeds; named bench property +
  named falsifier regime W11-4), but it is still **conditional**
  on (a) the bench property (delayed-causal-evidence with
  noise-corroborated decoy and specific-tier round-N
  disambiguation), (b) the closed-vocabulary generic-noise kind
  set being meaningful for the benchmark family, AND (c) round-N
  admission not being budget-starved (W11-4). Real multi-agent teams have additional axes
  (heterogeneous producers, time-varying budgets, multi-round
  handoffs, conflicting goals, generic-tier root_causes the
  bundle decoder cannot help with) the W10 family does not cover. The W4-2 result is proved-conditional
  (premises: faithful decoder + sound admission); the W4-C1 learned-
  policy advantage is conditional empirical-positive on the SDK v3.5
  config and falsified out-of-distribution on the SDK v3.7 real-LLM
  regime. External validity to real production multi-agent teams is
  partially advanced (three named regimes now anchored) but not
  fully closed.
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
  `docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md` (SDK v3.7),
  `docs/RESULTS_WEVRA_CROSS_ROLE_COHERENCE.md` (SDK v3.8),
  `docs/RESULTS_WEVRA_CROSS_ROLE_CORROBORATION.md` (SDK v3.9 — this milestone),
  `docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md` (SDK v3.9 — pre-committed bar)
- Paper draft: `papers/wevra_capsule_native_runtime.md`
- Tests: `vision_mvp/tests/test_wevra_capsule_native*.py`,
  `test_wevra_capsule_native_deeper.py`,
  `test_wevra_capsule_native_inner_loop.py` (SDK v3.4),
  `test_wevra_team_coord.py` (SDK v3.5 — multi-agent slice),
  `test_wevra_scale_vs_structure.py` (SDK v3.7 — Phase-53),
  `test_wevra_cross_role_coherence.py` (SDK v3.8 — Phase-54 + W7),
  `test_capsule_*.py`
- Cross-model parser-boundary experiment:
  `vision_mvp/experiments/parser_boundary_cross_model.py`
- Multi-agent team coordination benchmark (synthetic):
  `vision_mvp/experiments/phase52_team_coord.py`
- Stronger-model multi-agent benchmark (real LLM):
  `vision_mvp/experiments/phase53_scale_vs_structure.py`
- Cross-role cohort-coherence benchmark (deterministic):
  `vision_mvp/experiments/phase54_cross_role_coherence.py`
- Cross-role corroboration benchmark (deterministic, harder):
  `vision_mvp/experiments/phase55_decoy_plurality.py`
- MLX distributed runbook (operator path for Mac 2):
  `docs/MLX_DISTRIBUTED_RUNBOOK.md`
