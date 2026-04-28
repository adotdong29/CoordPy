# Research status — canonical, current

> Single-source-of-truth for the *active* research position of the
> Context Zero programme. If this file disagrees with any other
> doc on what is *true now*, this file is right and the other file
> is stale. For *theorem-by-theorem* status, see
> `docs/THEOREM_REGISTRY.md`. For *what may be claimed*, see
> `docs/HOW_NOT_TO_OVERSTATE.md`. Last touched: SDK v3.18,
> 2026-04-27.

## TL;DR

The programme now has **fourteen** coupled research axes, each
with a sharp status. SDK v3.18 mints axis 14: **magnitude-hinted
producer protocol + fresh-live end-to-end composition +
symmetric-corroboration limit theorem**. The W17 family adds one
new producer-prompt mode (``PRODUCER_PROMPT_MAGNITUDE_HINTED``),
one new dataclass (``OperationalThreshold``), one new schema
field, and one new prompt-render helper — purely additive on top
of the W14 surface. The runtime contract is byte-for-byte
unchanged.

**The headline SDK v3.18 results.** On a *fresh* live Mac-1
``qwen2.5:14b-32k`` Ollama probe at ``T_decoder = 14, K_auditor
= 8`` against the Phase-61 comparable-magnitude bank (n=8 × 24
producer calls; 0 endpoint failures; 128.2 s wall): under the
W17 magnitude-hinted prompt, bench property holds in **8/8**
(closing the 1/8 R-61-OLLAMA-A model-side judgment miss);
``capsule_attention_aware = 1.000``;
``capsule_layered_fifo_packed = 0.000``;
``capsule_fifo = 0.000``. **+1.000 strict separation** on both
axes — the **first programme result** that beats the strongest
non-composed baseline by ≥ 1.0 on a fresh live LLM probe (W17-1).

The W17-Λ-no-hint anchor on the same fresh probe under the
*legacy* structured prompt reproduces the W14-Λ-real envelope
(7/8 hold; +0.500 strict gain over FIFO-pack); the magnitude-
hint extension, not a re-run of the same prompt, is what closes
the gap from 0.500 to 1.000. The W17-Λ-naive falsifier on the
same probe under the naive prompt collapses to 0/8 + 0.000
(live counterpart of the W16-Λ-compose joint-failure regime).

The cross-model probe on a fresh live Mac-1 ``qwen3.5:35b`` MoE
backend (``think = False``; n=8 × 24 producer calls; 0 failures;
92.0 s wall) shows the magnitude-hint protocol **transfers**:
bench property holds in **8/8** (the W17 extension preserves the
bench-property hold-rate byte-for-byte across a 2.4× model-class
jump); ``capsule_attention_aware = 0.750``; **+0.750 strict gain**
over substrate FIFO and FIFO-packed-W14H-only. The 0.250 gap to
1.000 is on the ``accuracy_root_cause`` axis — a model-class-
specific specific-tier judgment artifact, not a producer-protocol
failure (W17-C-XMODEL, proved-conditional + empirical-research).

The **first explicit symmetric-corroboration limit theorem**
(W17-Λ-symmetric) lands as a *negative* result on the synthetic
``build_phase64_sym_bank`` (every service mentioned by exactly 2
distinct routed producer roles via generic-noise kinds with
comparable magnitudes; round-2 disambiguator names the gold
root_cause without a ``service=`` token). Under both
``T_decoder ∈ {None, 24}``: every capsule strategy in the SDK
ties FIFO at ``accuracy_full = 0.000`` — including the W14H +
W15 composition. The priority decoder still elects the right
specific-tier ``root_cause`` (``accuracy_root_cause = 1.000``);
the failure is ``services_correct`` set-equality. The structural
argument is that ``services_correct`` is an asymmetric oracle:
when the bipartite ``(role × tag, kind, magnitude)`` multiset is
symmetric for gold and decoy, no service-blind admission AND no
closed-form salience packer can prefer one. **W17-Λ-symmetric
discharges the prior W15-C-SYMMETRIC / W16-C-SYMMETRIC
conjectures as a negative theorem and names the next research
frontier**: a learned or LLM-distilled
semantic-disambiguator beyond the closed-form capsule surface
(W17-C-DISAMBIGUATOR, conjectural).

The **prior-conjecture discharge ledger** for SDK v3.18:
* W16-C-LIVE-OLLAMA → DISCHARGED (W17-1).
* W16-C-CROSS-MODEL → PARTIALLY DISCHARGED (W17-C-XMODEL on
  Ollama; MLX-distributed clause remains conjectural).
* W15-C-SYMMETRIC / W16-C-SYMMETRIC → DISCHARGED-NEGATIVE
  (W17-Λ-symmetric).

Backward-compat (W17-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``mode = 'naive'`` or ``mode = 'structured'``
AND ``magnitude_hinted_schema = False``, the W17 surface
reduces to the SDK v3.15 W14 anchor byte-for-byte. The Wevra
single-run product runtime contract is byte-for-byte unchanged.
See `docs/RESULTS_WEVRA_LIVE_COMPOSITION.md` for the milestone
note.

---

The *previous* (SDK v3.17) frontier mints axis 13: **end-to-end
W14 + W15 composition**. The composition does not introduce new
SDK code (no new admission policy, decoder, normaliser, packer,
or producer protocol — the runtime contract is byte-for-byte
unchanged). What SDK v3.17 *does* mint is the first programme
regime where the W14 producer-side ambiguity-preservation layer
AND the W15 decoder-side capsule context-packing layer are
*independently load-bearing on the same cell*.

On the new R-63-COMPOSED-TIGHT regime (Phase-62 multi-hypothesis
events × Phase-61 magnitude-filter producer simulation under the
W14 :class:`StructuredProducerProtocol` × ``T_decoder = 24``), the
W14+W15 composition achieves ``accuracy_full = 1.000`` while every
non-composed baseline collapses to 0.000 — a **+1.000 strict
separation** vs the W14-only-budgeted baseline (FIFO-packed-W13
over the structured-prompt stream) and the W15-only-without-W14
baseline (AttentionAwareBundleDecoder over the naive-prompt
stream). The result is stable across 5/5 alternate ``bank_seed``
values. The W16-Λ-compose joint-failure anchor on R-63-naive-tight
shows the producer-side limit (W14-Λ-prompt) and the decoder-side
limit (W15-Λ-budget) compose multiplicatively; closing one alone
leaves the other firing.

Crucially, the W16-Λ-real-replay anchor on the recorded Phase-61
``qwen2.5:14b-32k`` bytes (``phase61_real_ollama_structured_qwen2_5_14b_n8.json``,
n=8 × 24 producer calls, byte-stable, 0 endpoint failures) shows
the composed pipeline achieves ``capsule_attention_aware = 0.500``
while ``capsule_layered_fifo_packed = 0.000`` at ``T_decoder = 14``
on the recorded LLM bytes — a **+0.500 strict gain** over the
strongest non-composed baseline on a real-LLM stream. This is the
**first end-to-end real-LLM strict advance** in the programme that
beats the strongest non-composed baseline. The Mac-1 endpoint at
192.168.12.191:11434 was offline at milestone capture time
(``HTTP=000``), so a fresh live LLM probe (W16-C-LIVE-OLLAMA) is
conjectural.

Backward-compat (W16-3) preserved byte-for-byte: 442/442 prior
tests pass; with ``T_decoder = None`` and identity producer, the
composed pipeline ties the W13 layered decoder; with structured
prompt + ``T_decoder = None`` it ties the W14-1 anchor on R-61.
The Wevra single-run product runtime contract is byte-for-byte
unchanged. See `docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md` for the
milestone note.

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

## Current frontier (SDK v3.18, 2026-04-27)

### Active moves (SDK v3.18 — magnitude-hinted producer protocol + fresh-live end-to-end composition + symmetric-corroboration limit theorem + W17 family — *first fresh-live end-to-end real-LLM strict +1.000 advance + first explicit symmetric-corroboration negative theorem*)

- **Phase-64 fresh-live end-to-end composition + symmetric-
  corroboration wall benchmark.**
  ``vision_mvp.experiments.phase64_live_composition`` ships seven
  pre-committed sub-banks plus a falsifier and a cross-model cell:
  (i) ``r64_baseline`` — synthetic identity producer + magnitude-
  hinted prompt + ``T_decoder = None``. Sanity anchor: every
  cross-round capsule decoder hits 1.000.
  (ii) ``r64_w14h_only`` — synthetic mag-filter producer +
  magnitude-hinted prompt + ``T_decoder = None``. Synthetic
  counterpart of the W17-1 anchor. Bench property holds in 8/8;
  cross-round decoders hit 1.000 (W17-3 backward-compat).
  (iii) ``r64_live_struct`` — fresh live ``qwen2.5:14b-32k`` +
  legacy *structured* prompt + ``T_decoder = 14``. Reproduces the
  W14-Λ-real envelope on the fresh probe: bench property holds in
  7/8; ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.500 strict gain**.
  W17-Λ-no-hint structural anchor.
  (iv) ``r64_live_maghint`` — **the W17-1 anchor.** Same fresh
  live probe + magnitude-hinted prompt + ``T_decoder = 14``.
  Bench property holds in 8/8 (closing the 1/8 model-side miss);
  ``capsule_attention_aware = 1.000``;
  ``capsule_layered_fifo_packed = 0.000``;
  ``capsule_fifo = 0.000``; **+1.000 strict separation** on both
  axes. The first programme result that beats the strongest
  non-composed baseline by ≥ 1.0 on a fresh live LLM probe.
  (v) ``r64_live_naive`` — same fresh probe + naive prompt +
  ``T_decoder = 14``. Live counterpart of W14-Λ-prompt +
  W15-Λ-budget joint failure. Bench property holds in 0/8; every
  capsule strategy ties FIFO at 0.000. W17-Λ-naive falsifier.
  (vi) ``r64_live_xmodel`` — fresh live ``qwen3.5:35b`` MoE
  backend + magnitude-hinted prompt + ``T_decoder = 14``,
  ``think = False``. Cross-model probe (W17-C-XMODEL). Bench
  property holds in 8/8; ``capsule_attention_aware = 0.750``;
  ``capsule_layered_fifo_packed = 0.000``; **+0.750 strict gain**
  (well above the 0.50 strong-bar threshold). The 0.250 gap to
  1.000 is on ``accuracy_root_cause`` — a 35B-specific specific-
  kind judgment artifact, not a producer-protocol failure.
  Proved-conditional + empirical-research.
  (vii) ``r64_sym_loose`` and ``r64_sym_tight`` — synthetic
  symmetric-corroboration bank (every service mentioned by
  exactly 2 distinct routed producer roles via generic-noise
  kinds; round-2 disambiguator names gold root_cause without
  ``service=`` token); under both ``T_decoder ∈ {None, 24}``,
  every capsule strategy in the SDK ties FIFO at 0.000 — the
  **first explicit symmetric-corroboration limit theorem in the
  programme** (W17-Λ-symmetric).

- **W17 family minted.** W17-1 (proved-conditional +
  empirical-research; the first fresh-live end-to-end +1.000
  strict gain), W17-Λ-no-hint (empirical-research; live
  legacy-structured-prompt envelope), W17-Λ-naive (empirical-
  research; live joint-failure falsifier), **W17-Λ-symmetric**
  (proved-empirical + structural sketch; first explicit
  symmetric-corroboration limit theorem; *discharges*
  W15-C-SYMMETRIC / W16-C-SYMMETRIC as a negative theorem),
  W17-2 (proved + mechanically-checked; magnitude-hinted prompt
  determinism + threshold table soundness), W17-3 (proved-
  empirical full programme regression; the W17 surface reduces
  to the SDK v3.15 W14 anchor byte-for-byte under default
  parameters; 442/442 prior tests pass), **W17-C-XMODEL**
  (proved-conditional + empirical-research; fresh live 35B
  cross-model strict gain). The W17-C family (W17-C-DISAMBIGUATOR,
  W17-C-LEARNED-HINT, W17-C-CROSS-BENCH) makes the next research
  frontier explicit.

- **Magnitude-hinted producer protocol** (new SDK surface, purely
  additive). ``vision_mvp/wevra/team_coord.py`` ships:
  * ``PRODUCER_PROMPT_MAGNITUDE_HINTED`` — third producer-prompt
    mode; the W17-1 anchor.
  * :class:`OperationalThreshold` — closed-vocabulary record
    naming a kind, the qualifying field, the inclusive
    threshold, the unit, and a human gloss.
  * ``RoleExtractionSchema.magnitude_thresholds`` — additive
    optional field on the W14 schema; empty by default (W17-3
    byte-for-byte backward-compat); populated by
    ``incident_triage_role_schemas(magnitude_hinted=True)``.
  * :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` and
    :func:`incident_triage_magnitude_thresholds` — the
    pre-committed threshold table for the incident-triage family
    (calibrated to the synthetic
    :class:`MagnitudeFilteringExtractor`'s default thresholds,
    NOT to any specific scenario's magnitudes).
  * :func:`_render_magnitude_hinted_prompt` — the W17 prompt
    renderer. Adds the operational threshold table AND an
    anti-relative-magnitude clause to the structured prompt.

- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-64 anchor +
  bar 14 — live-end-to-end + magnitude-hinted-protocol +
  symmetric-corroboration-wall split + § 2.13 R-64 ingredients).
  The SDK v3.18 result clears the **strong success bar** § 1.1 on
  R-64-LIVE-MAGHINT (strict gain +1.000 vs both substrate FIFO
  AND FIFO-packed-W14H-only on a fresh live LLM probe; bench
  property held in 8/8 closes the prior 1/8 miss; named bench
  property + named falsifier regimes W17-Λ-no-hint /
  W17-Λ-naive / W17-Λ-symmetric AND named cross-model probe
  W17-C-XMODEL with +0.750 gain; W17-3 backward-compat
  preserved byte-for-byte). Headline data files:
  ``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``,
  ``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``,
  ``docs/data/phase64_cross_regime_synthetic.json``.

- **Honest scope.** The W17-1 win is *strongly conditional* on
  (a) the asymmetric-corroboration bench property, (b) the
  magnitude-hint table being calibrated to the synthetic
  extractor's threshold values (operational definitions, not
  answer hints — both gold and decoy magnitudes are well above
  every threshold), AND (c) the live endpoint reachable.
  W17-Λ-symmetric *names the wall* when the asymmetric ingredient
  is structurally absent: every capsule strategy in the SDK ties
  FIFO at 0.000 by construction, including the W14H + W15
  composition. The cross-model probe (W17-C-XMODEL) is
  proved-conditional on bench-property + strict-gain transfer; the
  saturated full-correctness clause remains conjectural and is
  bounded by a 35B-specific specific-tier judgment artifact (not
  a producer-protocol failure). The Wevra single-run product
  runtime contract is byte-for-byte unchanged.

### Active observations (SDK v3.18)

- **W17-Λ-real (honest positive on fresh live axis).** Real
  Mac-1 ``qwen2.5:14b-32k`` at ``temperature = 0`` on the
  Phase-61 redesigned comparable-magnitude events:
  - Under naive prompt: bench property holds 0/8; every method
    ties FIFO at 0.000 (live confirmation of W14-Λ-prompt +
    W15-Λ-budget joint failure on the live axis).
  - Under structured prompt: bench property holds 7/8;
    cross-round decoders all achieve ``accuracy_full = 0.500``;
    ``capsule_attention_aware - capsule_layered_fifo_packed
    = +0.500`` strict gain (W17-Λ-no-hint anchor — reproduces
    W14-Λ-real / W16-Λ-real-replay envelope on fresh probe).
  - Under magnitude-hinted prompt: bench property holds 8/8;
    cross-round decoders all achieve ``accuracy_full = 1.000``;
    ``capsule_attention_aware - capsule_layered_fifo_packed
    = +1.000`` strict gain (W17-1 anchor).
  - The 1/8 model-error failure on the slow_query_archival
    scenario (LLM judged decoy magnitudes "not severe enough"
    relative to gold) is **closed** by the W17 magnitude-hint
    extension; the LLM correctly emits every qualifying event
    once the absolute thresholds are stated and relative-
    magnitude skipping is forbidden in the prompt.

- **W17-C-XMODEL-real (proved-conditional cross-model
  bench-property transfer).** Real Mac-1 ``qwen3.5:35b`` MoE
  backend at ``temperature = 0`` with ``think = False``: bench
  property holds in 8/8 under the magnitude-hinted prompt (the
  bench-property hold-rate transfers to a 2.4× larger MoE
  model class byte-for-byte); ``capsule_attention_aware =
  0.750``; ``capsule_layered_fifo_packed = 0.000``; +0.750
  strict gain. The 0.250 gap to 1.000 is on
  ``accuracy_root_cause`` — the 35B emits a different specific-
  tier kind on one scenario than the 14B does (a model-side
  specific-kind judgment artifact, not a producer-protocol
  failure). The W5-1 (SDK v3.6) prior cross-model probe found
  parser-axis TVD = 1.000 between 14B and 35B under strict
  parsing; W17-C-XMODEL finds *bench-property-axis* TVD = 0.000
  under the magnitude-hinted prompt + robust parser — the W17
  protocol *normalises* the producer-side behaviour across the
  model-class jump that the W5-1 parser-boundary regime
  exposed.

### Active conjectures (SDK v3.18)

- **W17-C-DISAMBIGUATOR** (semantic-disambiguator beyond
  W17-Λ-symmetric): a learned or LLM-distilled disambiguator
  consuming the round-2 disambiguator's evidence text directly
  could distinguish ``orders_payments_join`` (gold A_B in
  deadlock) from a generic decoy whose round-1 mentions are
  observationally identical. **Conjectural; the named research
  frontier for SDK v3.19+.**

- **W17-C-LEARNED-HINT** (learned threshold variant): a per-kind
  data-driven threshold table outperforms the closed-form
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` on a
  held-out test set across 5/5 random splits. **Conjectural.**

- **W17-C-CROSS-BENCH** (cross-bench transfer): the magnitude-
  hint protocol generalises to non-incident-triage benchmark
  families when the family admits a closed-vocabulary kind →
  qualifying-threshold mapping. **Conjectural.**

### Discharged / partially-discharged conjectures (SDK v3.18)

- **W16-C-LIVE-OLLAMA** (SDK v3.17; "a fresh live Ollama probe
  closes the 1/8 model-error failure W14-only leaves").
  **DISCHARGED-empirical** by W17-1: a fresh live probe AT
  THE MAGNITUDE-HINTED PROMPT closes the 1/8 miss entirely
  (bench property holds in 8/8;
  ``accuracy_full = 1.000``); the W17-Λ-no-hint anchor on the
  same fresh probe under the legacy structured prompt
  reproduces the recorded 7/8 + 0.500 envelope, confirming the
  magnitude-hint extension (W17-1) is the load-bearing
  improvement, not a re-run of the same prompt.
- **W16-C-CROSS-MODEL** (SDK v3.17; "the composition's W16-1
  win at qwen2.5:14b-32k transfers to qwen3.5:35b under
  MLX-distributed inference"). **PARTIALLY DISCHARGED-
  empirical** by W17-C-XMODEL: the bench-property + strict-
  gain transfer is observed on the 35B Ollama backend (NOT
  MLX-distributed); the saturated full-correctness clause is
  still bounded by a 35B-specific judgment artifact; the
  MLX-distributed clause remains orthogonal and conjectural
  pending Mac-2 reachable.
- **W15-C-SYMMETRIC** / **W16-C-SYMMETRIC** (SDK v3.16 / v3.17;
  "symmetric-corroboration multi-hypothesis structural limit").
  **DISCHARGED as a negative theorem** by W17-Λ-symmetric: the
  R-64-SYM bench construction proves every capsule strategy in
  the SDK ties FIFO at 0.000 under both ``T_decoder ∈ {None,
  24}`` by construction. The wall is real and structural; the
  named research move beyond it is W17-C-DISAMBIGUATOR
  (conjectural).

## Previous frontier (SDK v3.17, 2026-04-27)

### Active moves (SDK v3.17 — end-to-end W14 + W15 composition + W16 family — *first end-to-end real-LLM strict advance over the strongest non-composed baseline*)

(See `docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md` for the SDK
v3.17 milestone note. SDK v3.18 supersedes this frontier on the
fresh-live and symmetric-wall axes; the W16 anchors are
preserved as historical evidence.)

## Previous frontier (SDK v3.16, 2026-04-27)

### Active moves (SDK v3.16 — attention-aware capsule context packing + W15 family — *first decoder-side context-packing move + joint-correctness-and-context-efficiency strict-gain anchor*)

- **Phase-62 attention-aware capsule context packing benchmark.**
  ``vision_mvp.experiments.phase62_attention_aware_packing`` ships
  three pre-committed sub-banks:
  (i) ``T_decoder=None`` — the R-62-default sanity anchor; every
  cross-round decoder (W11/W12/W13/W15) saturates at 1.000. The W15
  decoder ties W13 byte-for-byte on the answer field (W15-3
  backward-compat).
  (ii) ``T_decoder=24`` — the R-62-tightbudget W15-1 anchor; under
  FIFO packing the round-2 specific-tier disambiguator falls past
  the budget and ``capsule_layered_fifo_packed`` ties FIFO at
  ``accuracy_full = 0.000``. The W15
  :class:`AttentionAwareBundleDecoder` salience-packs with hypothesis
  preservation and achieves ``accuracy_full = 1.000``; +1.000 strict
  separation, stable across 5/5 alternate ``bank_seed`` values.
  (iii) ``T_decoder=9999`` — the R-62-saturation W15-Λ-budget
  falsifier; under no decoder-side budget pressure the salience
  reordering is a no-op on the answer field — both W15 and FIFO-pack
  tie at the W13 answer (1.000). The W15-1 win is *conditional* on
  budget pressure; this regime makes the conditionality sharp.
- **``AttentionAwareBundleDecoder`` + ``CapsuleContextPacker`` +
  ``FifoContextPacker`` (new).**
  ``vision_mvp/wevra/team_coord.py``. A two-stage decoder: (1)
  first-pass priority decode over the normalised union elects a
  tentative ``root_cause``; (2) salience-aware repack reorders +
  truncates the union under ``T_decoder`` (using closed-form weights
  on tier + CCK + corroboration + magnitude + round index, with
  ``preserve_hypotheses=True`` defaulting on) before final W13
  layered decode. Pack-stats expose ``position_of_first_causal_claim``
  (the proxy attention metric), ``tokens_kept_sum`` /
  ``tokens_input_sum``, ``hypothesis_count_kept``, and
  ``n_dropped_budget`` for direct audit. ``FifoContextPacker`` is the
  load-bearing baseline (FIFO truncation under the same
  ``T_decoder``). Re-exported via ``__all__``.
- **Theorem family W15.** W15-Λ-budget (decoder-side budget
  structural limit on R-62-tightbudget under FIFO packing,
  proved-empirical n=40 saturated × 5 seeds + structural sketch via
  W7-3 extension to the decoder-side axis),
  W15-1 (AttentionAwareBundleDecoder sufficiency under bounded
  ``T_decoder`` with hypothesis preservation, proved-conditional +
  proved-empirical synthetic n=40 saturated × 5 seeds, +1.000 vs
  fifo_packed_layered), W15-2 (pack determinism + closed-form
  salience, proved by inspection + mechanically-checked), W15-3
  (backward-compat with R-54..R-61 default banks, proved-empirical
  full programme-wide regression 393/393 + 37 new tests = 430/430),
  W15-Λ-degenerate (saturation falsifier on R-62-saturation,
  proved-empirical n=8: under no decoder-side budget pressure the
  W15-1 win is structurally invisible by construction), W15-4
  (token-efficiency floor: ``tokens_kept ≤ T_decoder`` strict, proved
  by inspection + mechanically-checked). The W15-C family (W15-C-real,
  W15-C1, W15-C-LEARNED, W15-C-SYMMETRIC, W15-C-COMPOSE-W14) makes
  real-LLM-downstream-decoder, cross-bench, learned-salience,
  symmetric-corroboration, and W14+W15 compose extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-62 anchor +
  bar 12 — joint-correctness-and-context-efficiency split + § 2.11
  R-62 ingredients). The SDK v3.16 result clears the **strong
  success bar** § 1.1 on R-62-tightbudget synthetic (strict gain
  +1.000 vs FIFO-packed-W13, stable across 5/5 (bank_seed) values,
  no regression on R-53..R-61, audit T-1..T-7 preserved on every
  cell, named bench property + named falsifier regime
  W15-Λ-degenerate, AND joint-correctness-and-context-efficiency
  split bar 12 satisfied — the new method includes a load-bearing
  decoder-side context-packing intervention beyond every prior
  layer). Headline data file:
  ``docs/data/phase62_seed_sweep_tightbudget_K12_n8.json``.
- **Honest scope.** R-62 is a *synthetic* milestone — the producer
  is :class:`IdentityExtractor`, not a real LLM. Real-LLM transfer
  of W15 is W15-C-real, conjectural; it requires Mac 1 / Mac 2 to
  be online and the bundle to be re-decoded by an LLM agent under a
  real context window. SDK v3.16 does not run this probe.
  "Attention-aware" uses an *honest proxy* — the
  ``position_of_first_causal_claim`` metric — not transformer
  attention manipulation. The W15-1 win is *conditional* on (a) the
  bench property holding, (b) ``T_decoder`` below the union token
  sum, AND (c) round-2 carrying a specific-tier disambiguator with
  no ``service=`` token; W15-Λ-degenerate makes the conditionality
  sharp. The Wevra single-run product runtime contract is byte-for-
  byte unchanged.

### Prior moves (SDK v3.15 — structured producer protocol + W14 family — *first producer-protocol move + first real-LLM strict gain ≥ 0.50 over substrate FIFO*)

- **Phase-61 producer-side ambiguity-preservation benchmark.**
  ``vision_mvp.experiments.phase61_producer_ambiguity_preservation``
  ships three pre-committed sub-banks plus an opt-in real-Ollama
  extension:
  (i) ``identity`` — the R-61-default sanity anchor; the synthetic
  extractor passes every emission through verbatim. Bench property
  holds in 8/8; W11/W12/W13 cross-round decoders all hit 1.000.
  (ii) ``magnitude_filter`` + ``naive`` prompt — the W14-Λ-prompt
  anchor; the magnitude-filter extractor calibrated against the
  W13-Λ-real real-Ollama observation collapses round-1 by top-N-
  per-(role, kind) by magnitude. On the redesigned comparable-
  magnitude events the threshold filter does not fire (the events
  are designed to clear it) but the prompt-induced compression
  drops monitor's decoy-side mentions; the bench property holds in
  0/8 and every capsule strategy ties FIFO at 0.000.
  (iii) ``magnitude_filter`` + ``structured`` prompt — the W14-1
  anchor; the structured prompt's per-event mandate disables the
  compression. Bench property holds in 8/8; the cross-round capsule
  pipeline achieves ``accuracy_full = 1.000``; +1.000 strict
  separation vs the naive-prompt counterpart, stable across 5/5
  alternate ``bank_seed`` values.
  (iv) ``ollama`` (opt-in) — the R-61-ollama probe; talks Mac-1
  ``qwen2.5:14b-32k`` with the structured prompt; raw producer
  responses captured into the report. **R-61-OLLAMA-A tier** (the
  strong-real-transfer tier of `docs/SUCCESS_CRITERION_MULTI_AGENT_
  CONTEXT.md` § 1.5): bench property holds in 7/8 scenarios;
  cross-round decoders achieve ``accuracy_full = 0.500``;
  ``layered − fifo = +0.500`` at exactly the 0.50 threshold; audit
  T-1..T-7 preserved on every cell. Anchor:
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``.
- **``StructuredProducerProtocol`` + ``RoleExtractionSchema`` (new).**
  ``vision_mvp/wevra/team_coord.py``. A prompt-rendering surface
  with two modes: ``naive`` (legacy Phase-58/59/60 byte-for-byte)
  and ``structured`` (W14: tier banner observation/diagnosis split +
  per-tier kind whitelist + per-event mandate). The protocol
  consumes a deterministic ``RoleExtractionSchema`` (allowed kinds
  + observation/diagnosis partition); ``incident_triage_role_
  schemas()`` ships the schema table for the Phase-58..Phase-61
  bench family. A new structured-aware response parser
  (``_parse_structured_response``) dedupes by ``(kind, payload)``
  rather than ``kind`` alone so the per-event mandate survives
  parsing. Re-exported via ``__all__``.
- **Theorem family W14.** W14-Λ-prompt (producer-side ambiguity-
  erasure structural limit on R-61-naive-prompt, proved-empirical
  n=40 saturated × 5 seeds + structural sketch via W7-3 extension),
  W14-1 (StructuredProducerProtocol sufficiency under bounded
  producer compression, proved-conditional + proved-empirical
  synthetic n=40 + real Ollama n=8), W14-2 (schema soundness +
  protocol determinism, proved by inspection + mechanically-
  checked), W14-3 (backward-compat with R-54..R-60, proved-empirical
  full programme-wide regression 393/393), W14-4 (combined-
  intervention falsifier on R-61-ollama-naive, proved-empirical
  n=8), W14-Λ-real (real Ollama 14B prompt-protocol transfer,
  empirical-research n=8 × 24 producer calls). The W14-C family
  (W14-C1..W14-C5) makes cross-bench, model-side calibration,
  multi-round generalisation, cross-model transfer, and multi-
  hypothesis variant extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-61 anchor +
  bar 11 — producer-side ambiguity-preservation split + § 1.5
  R-61-ollama 4-tier grading). The SDK v3.15 result clears the
  **strong success bar** § 1.1 on R-61-structured-prompt synthetic
  (strict gain ≥ 0.20 vs every prior anchor including SDK v3.14
  W13 alone, stable across ≥ 3 (bank_seed) values, no regression on
  R-53..R-60, audit T-1..T-7 preserved on every cell, named bench
  property + named falsifier regime W14-4, AND producer-side
  ambiguity-preservation split bar 11 satisfied — the new method
  includes a load-bearing prompt-side intervention beyond every
  prior downstream layer). The R-61-ollama probe lands at the
  **R-61-OLLAMA-A tier** (the strongest real-transfer tier): the
  milestone is therefore a *strong-success on R-61 synthetic + a
  strong-success / OLLAMA-A on R-61-ollama*, the **first strong-
  success on a real-LLM stream in the programme** (R-60-OLLAMA-C
  was honest null; R-61-OLLAMA-A is +0.500 strict gain at the
  threshold).
- **Honest scope.** The W14-1 win is *conditional* on (a) the
  redesigned comparable-magnitude events (R-61 closure property),
  (b) the structured producer protocol being in effect, AND (c) the
  cross-round capsule pipeline (admission + multi-round decoder).
  The W14-4 falsifier regime (real Ollama + comparable-magnitude
  events + naive prompt) is the explicit counterexample showing
  *both* interventions are required. The W13 closure-widening is
  structurally invisible on R-61-ollama because the real LLM emits
  canonical kinds (zero drift); on the structured prompt the
  load-bearing layer is W14, not W13. The W13 layer is dormant on
  this regime, not refuted.

### Active observations (SDK v3.15)

- **W14-Λ-real (honest positive).** Real Mac-1
  ``qwen2.5:14b-32k`` at ``temperature=0`` on the Phase-61
  redesigned comparable-magnitude events:
  - Under naive prompt: bench property holds 0/8; every method
    ties FIFO at 0.000 (empirical confirmation of W14-Λ-prompt on
    real LLM).
  - Under structured prompt: bench property holds 7/8; W11/W12/W13
    cross-round decoders all achieve ``accuracy_full = 0.500``;
    +0.500 strict gain vs FIFO at the R-61-OLLAMA-A threshold.
  - The 1/8 model-error failure is on the LLM (slow_query
    scenario: LLM judged ``error_rate=0.15`` not to qualify as
    ``ERROR_RATE_SPIKE``), not on the protocol — the structured
    prompt + comparable-magnitude events restore the bench
    property in every other scenario. The W14-C2 conjecture
    (magnitude-hinted prompt extension) is the natural next move.

### Active conjectures (SDK v3.15)

- **W14-C1**: cross-bench transfer of the W14 protocol to non-
  incident-triage benchmark families. Conjectural.
- **W14-C2**: model-side magnitude calibration via a *magnitude
  hint* extension to the structured prompt. Conjectural; the W14-
  Λ-real 7/8 anchor is the candidate falsifier.
- **W14-C3**: multi-round generalisation to N ≥ 3 rounds with a
  graded tier hierarchy. Conjectural.
- **W14-C4**: cross-model transfer to qwen3.5:35b-MoE and to non-
  Ollama backends (MLX-distributed). Conjectural; requires Mac 2
  reachable.
- **W14-C5**: multi-hypothesis variant of the protocol that permits
  2-3 candidate kinds per event. Conjectural.

### Discharged / partially-discharged conjectures (SDK v3.15)

- **W13-C3** (SDK v3.14; real-Ollama transfer of W13-1 with a
  redesigned event stream + a prompt that instructs the LLM to
  emit one claim per distinct event). **PARTIALLY DISCHARGED-
  empirical** by W14-1 + W14-Λ-real: the *combination* of
  comparable-magnitude events + structured producer protocol
  clears the strict-gain bar on real Ollama 14B (n=8, +0.500 vs
  FIFO at exactly the R-61-OLLAMA-A threshold), BUT the gain is on
  the **W11/W12/W13 cross-round pipeline as a whole** — not
  specifically on the W13 closure-widening, which is structurally
  invisible because the real LLM still emits canonical kinds. The
  honest revised reading: prompt-side discipline + comparable-
  magnitude events restore the bench property; W13's *normalisation*
  contribution remains dormant on this regime. The cross-round
  structural bar is cleared at +0.500 by the combined intervention
  (W14).

## Previous frontier (SDK v3.14, 2026-04-26)

### Active moves (SDK v3.14 — layered open-world normalisation + real-Ollama probe + W13 family — *first open-world normalisation move and first real-Ollama transfer measurement*)

- **Phase-60 open-world normalisation + real-Ollama benchmark.**
  ``vision_mvp.experiments.phase60_open_world_normalization``
  ships three pre-committed sub-banks plus an opt-in real-Ollama
  extension:
  (i) ``synthetic_wide_oov_llm`` — the W13-1 anchor; the producer
  emits drifted variants from
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` (every entry verified
  *outside* :data:`CLAIM_KIND_SYNONYMS` and *inside* the W13
  heuristic closure); pre-committed default
  ``K_auditor=8, T_auditor=256, n_eval=12, bank_seed=11,
  wide_oov_prob=0.50, svc_token_alt_prob=0.30``. Headline: ``capsule_layered_multi_round`` achieves
  ``accuracy_full = 1.000`` while ``capsule_robust_multi_round``
  (W12) **and** every other strategy ties FIFO at 0.000 — the
  *first strict separation between layered (open-world) and
  fixed-vocabulary normalisation* in the programme; +1.000 vs every
  un-widened method including SDK v3.13 W12; stable across **5/5**
  alternate (bank_seed, llm_seed) values with min gap layered − w12
  = +0.917 (well above the 0.50 stability threshold).
  (ii) ``synthetic_cosmic_oov_llm`` — the W13-4 closure-boundary
  anchor; round-2 specific-tier kinds are replaced by entries from
  :data:`COSMIC_OOV_KINDS` (every entry verified outside both
  layers); ``capsule_layered_multi_round`` ties FIFO at 0.000 on
  8/8 falsifier scenarios.
  (iii) ``synthetic_clean_llm`` + cross-regime audit — the W13-3
  backward-compat anchor; on R-58 / R-59 / R-60-clean / R-54..R-57,
  the W13 decoder produces byte-for-byte the same answer as W11 /
  W12 (heuristic layer fires zero times); R-54..R-58 / R-59-noisy
  anchors all preserved at ``accuracy_full = 1.000``.
  (iv) ``ollama`` — the R-60-ollama probe; talks Mac-1
  ``qwen2.5:14b-32k`` on the same Phase-58 events; raw producer
  responses captured into the report. **Honest negative finding
  (W13-Λ-real, see § *Active observations* below)**: real Ollama 14B
  emits canonical kinds and filters low-magnitude decoy events;
  the bench property holds in 0/4 scenarios; W13 has nothing to
  rescue and ties W12.
- **``LayeredClaimNormalizer`` + ``LayeredRobustMultiRoundBundleDecoder``
  (new).** ``vision_mvp/wevra/team_coord.py``. Two-layer normaliser:
  (a) exact :data:`CLAIM_KIND_SYNONYMS` lookup (the W12 path); (b)
  ordered :data:`_HEURISTIC_KIND_RULES` regex-predicate abstraction
  rules whose union strictly widens the W12 closure; (c) optional
  abstention via the :data:`LAYERED_NORMALIZER_ABSTAIN` sentinel.
  Per-call counters expose load-bearing layer breakdowns
  (``n_exact``, ``n_heuristic``, ``n_abstained``, ``n_passthrough``,
  ``rule_hits``). Re-exported as ``LayeredClaimNormalizer``,
  ``LayeredRobustMultiRoundBundleDecoder``,
  ``HeuristicAbstractionRule``, ``LAYERED_NORMALIZER_ABSTAIN``.
- **Theorem family W13.** W13-Λ-fixed (fixed-vocabulary closure
  limit on R-60-wide, proved-empirical n=12 + structural sketch),
  W13-1 (LayeredRobustMultiRoundBundleDecoder sufficiency under
  bounded OOV in the heuristic closure, proved-conditional + proved-
  empirical n=60 saturated across 5 seeds), W13-2 (heuristic
  abstraction soundness, proved by inspection + mechanically-
  checked), W13-3 (backward-compat with R-54..R-58 + R-59 + R-60-
  clean, proved-empirical n=8 each + cross-regime audit), W13-4
  (cosmic-OOV closure boundary, proved-empirical n=8 saturated),
  W13-Λ-real (real Ollama 14B canonical-kind + magnitude-filtering
  observation, empirical-research n=4 producer-side observation +
  12 real Ollama calls). The W13-C family (W13-C1..W13-C4) makes
  cross-bench, learned-normaliser, real-Ollama-with-redesigned-
  events, and abstention-aware-decoder extensions falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-60 anchor +
  bar 10 — open-world normalisation split + § 1.4 R-60-ollama
  4-tier grading). The SDK v3.14 result clears the **strong success
  bar** § 1.1 on R-60-wide (strict gain ≥ 0.20 vs every fixed-
  vocabulary method including SDK v3.13 W12, stable across ≥ 3
  (bank_seed, llm_seed) values, no regression on
  R-53..R-59 / R-60-clean, audit T-1..T-7 preserved on every cell,
  named bench property + named falsifier regime W13-4, AND open-
  world normalisation split bar 10 satisfied — the new method
  includes a load-bearing heuristic abstraction layer beyond the
  exact synonym table). The R-60-ollama probe lands at the
  **R-60-OLLAMA-C tier** (honest null real transfer): the milestone
  is therefore a *strong-success on R-60-wide synthetic + a
  partial-success / honest-null on R-60-ollama*, NOT a strong
  cross-axis advance. § 1.4 of the success criterion makes this
  partition pre-committed and falsifiable.
- **Honest scope.** The W13-1 win is *conditional* on (a) the named
  bench property (R-58 delayed-causal-evidence shape with the
  Phase-60 wide-OOV drift channel), (b) the producer-noise channel
  being bounded by the *heuristic* closure (every variant in
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` matches at least one
  pattern in :data:`_HEURISTIC_KIND_RULES`), AND (c) round-N
  admission not being budget-starved (inherits W11-4). The W13-4
  falsifier regime is the explicit counterexample; the W13-Λ-real
  observation is a *separate, honest, partial* outcome. The W13
  method is research-grade SDK code, additive on top of W12.

### Active observations (SDK v3.14)

- **W13-Λ-real (honest negative).** Real Ollama 14B
  (qwen2.5:14b-32k on Mac 1, ``temperature=0``) on the calibrated
  Phase-58 incident-triage prompt does NOT generate the R-58
  delayed-causal-evidence bench property: across 4 scenarios × 12
  producer calls, the LLM emits 0 drifted kinds (every claim_kind
  is canonical) and filters low-magnitude decoy events as noise
  (the ``monitor`` role emits ``NONE`` for the deliberately-low-
  magnitude decoy events, breaking the cross-role decoy
  corroboration assumption). The bench property holds in 0/4
  scenarios; normalisation has nothing to rescue; W13 ties W12 ties
  multi_round at ``accuracy_full = 0.250``. The R-60-ollama probe
  is therefore a *measurement*, not a *claim*: the synthetic→real-
  LLM transfer story has **five layers** —
  (i) un-normalised admission cannot transfer (W6-C2 falsified),
  (ii) un-normalised cross-round decoding cannot transfer
  (W12-Λ at the real-LLM axis),
  (iii) fixed-vocabulary normalisation transfers under bounded
  *synthetic* drift (W12-1, conditional),
  (iv) heuristic-widened normalisation transfers under bounded
  *open-world* drift inside the heuristic closure (W13-1,
  conditional),
  (v) real Ollama 14B at default settings does not produce the
  drift OR the cross-role decoy corroboration shape (W13-Λ-real,
  empirical observation; the gating axis on real Ollama is *event-
  shape design + prompt-side discipline*, not normalisation).
  Future work: redesign the events so the decoy has comparable
  magnitudes to gold (W13-C3) — and accept that the contribution
  shifts from "the normaliser" to "the prompt + event design".

### Active conjectures (SDK v3.14)

- **W13-C1**: cross-bench transfer of the W13 closure-widening
  contract to non-incident-triage benchmark families.
  Conjectural; falsifier = a benchmark family where any size-
  bounded predicate set covers < 50 % of LLM kind drift.
- **W13-C2**: a learned normaliser strictly widens the W13
  heuristic closure on R-60-cosmic. Conjectural; restated as a
  closure-widening move, not a structural fix.
- **W13-C3**: real-Ollama transfer of W13-1 with redesigned
  events. Conjectural; Phase-60 v2 candidate.
- **W13-C4**: abstention as a load-bearing signal — an
  abstention-aware decoder strictly improves over a passthrough
  decoder. Conjectural; the abstention sentinel is implemented but
  the abstention-aware decoder is not yet wired.

### Discharged / partially-discharged conjectures (SDK v3.14)

- **W12-C2** (SDK v3.13; real-Ollama transfer of W12-1).
  **PARTIALLY DISCHARGED-empirical** (negatively): real Ollama 14B
  on the Phase-58 events does NOT emit drift, so the W12 advance
  is *structurally invisible* on R-60-ollama (W13-Λ-real). The
  W12-C2 question reframes as: under what (event design × prompt)
  does a real LLM emit non-trivial bounded drift? — that is W13-C3.
- **W12-C3** (SDK v3.13; learned synonym table widens the W12
  closure). **PARTIALLY DISCHARGED-empirical** by W13-1: a
  *heuristic* layered normaliser strictly widens the closure
  beyond the fixed table on R-60-wide. The *learned* variant
  remains conjectural (W13-C2). The W13-4 closure boundary applies
  to any predicate-based normaliser, learned or not — a learned
  normaliser only widens, does not eliminate, the closure.

## Previous frontier (SDK v3.13, 2026-04-26)

### Active moves (SDK v3.13 — real-LLM-robust multi-round bundle decoder + W12 family — *first synthetic→real-LLM transfer move*)

- **Phase-59 real-LLM-driven multi-round delayed-disambiguation
  benchmark.**
  ``vision_mvp.experiments.phase59_real_llm_multi_round`` runs 10
  strategies (substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_cohort_buffered (W7-2),
  capsule_corroboration (W8), capsule_multi_service (W9),
  capsule_bundle_decoder (W10 single-round), capsule_multi_round
  (W11), capsule_robust_multi_round (W12)) on the same Phase-58
  delayed-causal-evidence shape under an **LLM-shaped extractor**:
  default mode ``synthetic_noisy_llm`` injects realistic
  claim-kind drift (``synonym_prob=0.50``) and payload drift
  (``svc_token_alt_prob=0.30``) calibrated against Phase-53
  empirical 14B / 35B parser_role_response distributions; the
  optional ``ollama`` mode talks the same Phase-53-style prompt
  to a real LLM endpoint with synthetic fallback on HTTP failure.
  12/12 default scenarios satisfy the post-normalisation
  delayed-causal-evidence property; mechanically verified by
  ``Phase59BenchPropertyTests``. Pre-committed default:
  ``K_auditor=8``, ``T_auditor=256``, ``n_eval=12``,
  ``bank_seed=11``, ``bank_replicates=3``,
  ``llm_synonym_prob=0.50``, ``llm_svc_alt_prob=0.30``,
  ``llm_seed=11``. Headline: ``capsule_robust_multi_round``
  achieves ``accuracy_full = 1.000`` while substrate, FIFO,
  priority, coverage, W7-2, W8, W9, W10 single-round bundle, AND
  **W11 un-normalised** all produce ``accuracy_full = 0.000`` —
  the **first strict separation between un-normalised and
  normalised cross-round capsule-native coordination on a
  real-LLM-shaped stream** in the programme, **+1.000** vs every
  other method, stable across **5/5** alternate (bank_seed,
  llm_seed) values. The W12 family (W12-Λ / W12-1 / W12-2 /
  W12-3 / W12-4 — proved or proved-empirical) anchors the
  milestone formally; the W12-C family (W12-C1/C2/C3) makes the
  cross-bench / real-Ollama / learned-normaliser extensions
  falsifiable.
- **``RobustMultiRoundBundleDecoder`` (new).**
  ``vision_mvp/wevra/team_coord.py``. Wraps
  :class:`MultiRoundBundleDecoder` with a closed-vocabulary
  normalisation layer: :func:`normalize_claim_kind` rewrites
  drifted ``claim_kind`` strings into canonical kinds via
  :data:`CLAIM_KIND_SYNONYMS` (≈ 60 entries covering 11 canonical
  kinds × 4-5 LLM variants each, lex-ordered for diff stability);
  :func:`normalize_payload` rewrites alternative service-tag
  spellings (``svc=X``, ``for service X``, ``service:X``,
  ``service_name=X``, …) into the canonical ``service=<tag>`` form
  via the closed-vocabulary :data:`_SERVICE_TAG_REWRITES` regex
  table. Per-call rewrite counters (``last_n_kind_rewrites``,
  ``last_n_payload_rewrites``) expose the normaliser's load-bearing
  status to the bench driver. Re-exported as
  ``RobustMultiRoundBundleDecoder``, ``CLAIM_KIND_SYNONYMS``,
  ``normalize_claim_kind``, ``normalize_payload``,
  ``normalize_handoff``.
- **Theorem family W12.** W12-Λ (real-LLM single-round / un-
  normalised structural limit on R-59, proved-empirical n=12 + 5
  seeds + structural sketch), W12-1 (RobustMultiRoundBundleDecoder
  sufficiency under bounded LLM noise, proved-conditional + proved-
  empirical n=60 saturated across 5 seeds), W12-2 (closed-
  vocabulary normalisation soundness, proved by inspection +
  mechanically-checked), W12-3 (backward-compat with R-58 + R-59-
  clean + cross-regime R-54..R-58 audit, proved-empirical n=8
  each), W12-4 (out-of-vocabulary noise-budget falsifier, proved-
  empirical n=8 saturated). The W12-C family (W12-C1/C2/C3) makes
  the cross-bench / real-Ollama / learned-normaliser extensions
  falsifiable.
- **Pre-committed success criterion** in
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-59 anchor +
  bar 9 — synthetic→real-LLM transfer split). The SDK v3.13 result
  clears the **strong success bar** § 1.1 (strict gain ≥ 0.20 on
  R-59 vs every un-normalised single-round / multi-round method
  including SDK v3.12 W11, stable across ≥ 3 (bank_seed, llm_seed)
  values, no regression on R-53 / R-54 / R-55 / R-56 / R-57 /
  R-58 / R-59-clean, audit T-1..T-7 preserved on every cell, named
  bench property + named falsifier regime, AND synthetic→real-LLM
  transfer split bar 9 satisfied — the new method includes the
  load-bearing closed-vocabulary normalisation layer that bar 9
  requires).
- **Honest scope.** The W12-1 win is *conditional* on (a) the
  named bench property (R-58 delayed-causal-evidence shape), (b)
  the producer-noise channel being bounded by the closed-vocabulary
  closure (every variant in :data:`NOISY_KIND_VARIANTS` is in
  :data:`CLAIM_KIND_SYNONYMS`), AND (c) round-N admission not being
  budget-starved (inherits W11-4). The W12-4 falsifier regime is
  the explicit counterexample: when the LLM emits *out-of-vocabulary*
  kinds the synonym table cannot cover (e.g.
  ``DEADLOCK_PROBABLY_DETECTED_MAYBE``), normalisation cannot
  rescue the run. The synthetic-noisy-LLM extractor is *labelled*
  in every Phase-59 report; the ``ollama`` opt-in mode is the
  honest extension path and is the W12-C2 next data point.

### Active conjectures (SDK v3.13)

- **W12-C1**: cross-bench transfer of the W12 normalisation
  contract to non-incident-triage benchmark families.
  Conjectural; falsifier = a benchmark family where LLM kind
  drift cannot be enclosed by any reasonable-size synonym table.
- **W12-C2**: real-Ollama transfer of W12-1 (Phase-59 ``ollama``
  mode against qwen2.5:14b-32k or qwen3.5:35b on Mac 1).
  Conjectural; the synthetic noisy channel is calibrated to the
  empirical Phase-53 14B/35B distributions, but the real LLM may
  emit drift outside the closed-vocabulary closure.
- **W12-C3**: a learned normaliser strictly widens the closure
  beyond the hand-curated table. Conjectural; restated as a
  research move not a structural fix.

### Discharged conjectures (SDK v3.13)

- **W11-C2** (SDK v3.12): real-LLM transfer of W11-1.
  **PARTIALLY DISCHARGED-empirical** by the W12 family: the
  *un-normalised* W11 decoder does NOT transfer (W12-Λ shows
  multi_round ties FIFO at 0.000 on Phase-59 default at
  ``synonym_prob=0.50``), but a *normalised* W11 decoder
  (W12-1) DOES transfer (+1.000 vs every un-normalised method,
  stable 5/5). The honest revised reading: synthetic cross-round
  structure transfers to real-LLM regimes *only when an explicit
  normalisation layer absorbs the producer's kind / payload drift
  channel*.

## Previous frontier (SDK v3.12, 2026-04-26)

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
- **Not** "we solved multi-agent context." SDK v3.14's W13-1 result
  is the strongest *open-world-drift-under-bounded-heuristic-
  closure* structural-win the programme has produced
  (LayeredRobustMultiRoundBundleDecoder wins on R-60-wide by +1.000
  vs every fixed-vocabulary method **including SDK v3.13 W12**;
  backward-compatible on R-54..R-58 / R-59 / R-60-clean; stable
  across 5/5 (bank_seed, llm_seed) values; named bench property +
  named falsifier regime W13-4), but it is still **conditional on**
  (a) the bench property (R-58 delayed-causal-evidence shape with
  the Phase-60 wide-OOV drift channel), (b) the producer-noise
  channel being bounded by the heuristic closure, AND (c) round-N
  admission not being budget-starved (inherits W11-4). On a real
  Ollama producer at the 14B class, the bench property does NOT
  hold by default (W13-Λ-real); the synthetic→real-LLM transfer is
  gated by event-shape design + prompt-side discipline, not by
  normalisation. This is an honest empirical finding, not a closure
  of the question.

  SDK v3.13's W12-1 result remains the strongest *real-LLM-shaped-
  stream* (synthetic noisy) structural-win the programme has
  produced (RobustMultiRoundBundleDecoder wins on
  R-59 by +1.000 vs every un-normalised single-round / multi-round
  method **including SDK v3.12 W11**; backward-compatible on
  R-54 / R-55 / R-56 / R-57 / R-58 / R-59-clean; stable across
  5/5 (bank_seed, llm_seed) values; named bench property + named
  falsifier regime W12-4), but it is still **conditional** on
  (a) the bench property (R-58 delayed-causal-evidence shape),
  (b) the producer-noise channel being bounded by the closed-
  vocabulary closure (every variant in :data:`NOISY_KIND_VARIANTS`
  is in :data:`CLAIM_KIND_SYNONYMS`), AND (c) round-N admission
  not being budget-starved (inherits W11-4). The synthetic-noisy-
  LLM extractor is calibrated against Phase-53 14B/35B empirical
  distributions; the ``ollama`` opt-in mode is the W12-C2 next
  data point. Real multi-agent teams have additional axes
  (heterogeneous producers, time-varying budgets, multi-round
  handoffs with > 2 rounds and inter-round contradictions,
  conflicting goals, generic-tier root_causes the bundle decoder
  cannot help with, OOV kinds outside any reasonable closure) the
  W12 family does not cover. The W4-2 result is proved-conditional
  (premises: faithful decoder + sound admission); the W4-C1 learned-
  policy advantage is conditional empirical-positive on the SDK v3.5
  config and falsified out-of-distribution on the SDK v3.7 real-LLM
  regime. External validity to real production multi-agent teams is
  *materially* advanced by SDK v3.13 (the first synthetic→real-LLM
  transfer move with a named bounded-noise channel) but not fully
  closed.
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
