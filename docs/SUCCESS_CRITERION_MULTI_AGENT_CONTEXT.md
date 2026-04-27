# Success criterion — solving multi-agent context (SDK v3.14 bar)

> Pre-committed, falsifiable bar for what counts as a *real* advance
> on "solving multi-agent context" in the Context Zero / Wevra
> programme. This document is the **referee** for SDK v3.9 / v3.10 /
> v3.11 / v3.12 / v3.13 / v3.14 (and later milestones). Any milestone
> note that claims an advance must name the bar it cleared and cite
> the code-anchored evidence. Last touched: SDK v3.14, 2026-04-26.
>
> The history of this programme is full of moves where a partial
> result was written up too strongly and later had to be sharpened
> or retracted (W3-C7 strict, W4-C1 OOD, W6-C1, W6-C2). This bar
> is the dam against more of the same.

## TL;DR

A milestone *advances* the multi-agent-context thesis only if it
clears one of three pre-committed bars below. A milestone *does
not* advance the thesis if it produces a single cherry-picked
cell, regresses on a previously-shipped regime, or breaks the
lifecycle audit on any cell. Honest negative or null results are
acceptable — but they must be *labelled* as such, not framed as
advances.

The **named regimes** the bar refers to (anchored in code):

* **R-53** — Phase-53 real-LLM regime (low-surplus, single-service
  per scenario). FIFO unbeatable by W7-1.
  Anchor: `vision_mvp/experiments/phase53_scale_vs_structure.py`.
* **R-54** — Phase-54 deterministic gold-plurality regime
  (foreign-service decoys, gold tag in strict raw plurality).
  W7-2 win.
  Anchor:
  `vision_mvp/experiments/phase54_cross_role_coherence.py`.
* **R-55** — Phase-55 decoy-plurality + cross-role-corroborated-gold
  regime. Single-service-gold (`gold_services` of size 1).
  W8 win.
  Anchor:
  `vision_mvp/experiments/phase55_decoy_plurality.py`.
* **R-56** — Phase-56 *multi-service-gold + cross-role-corroborated*
  regime, the *new harder regime* SDK v3.10 introduces. Each
  scenario has `gold_services` of size 2, both gold services
  corroborated by ≥ 2 distinct producer roles. The W8 single-tag
  policy admits only the top-1 corroborated tag and fails the
  decoder's set-equality `services_correct` check; W9
  (`MultiServiceCorroborationAdmissionPolicy`, top-K with
  argmax-by-role-count gate) admits all tags in the cross-role-
  corroboration argmax tier and recovers the multi-service gold.
  Anchor:
  `vision_mvp/experiments/phase56_multi_service_corroboration.py`.
* **R-57** — Phase-57 *multi-service-gold + corroborated-decoy
  decoder-forcing* regime, the *new harder regime* SDK v3.11
  introduces. Multi-service gold (`|gold_services| = 2`), both gold
  services corroborated by ≥ 2 distinct producer roles via
  *root-cause-causal* claim_kinds, AND a decoy service is *also*
  corroborated by ≥ 2 distinct producer roles via *non-causal*
  generic-noise claim_kinds (`LATENCY_SPIKE`, `ERROR_RATE_SPIKE`,
  `FW_BLOCK_SURGE`). The decoy enters W9's argmax tier; W9 picks
  top-K by score and either drops a gold service (when decoy beats
  one gold on raw count) or — under top_k≥3 — admits the decoy and
  the decoder's set-equality check still fails. **No
  service-blind admission policy can beat FIFO on R-57 by
  construction (W10-Λ).** A *bundle-aware decoder* that filters
  the admitted services by the chosen `root_cause`'s causal
  claim-kind set (`CCK`) closes the gap (W10-1). Anchor:
  `vision_mvp/experiments/phase57_decoder_forcing.py`.
* **R-58** — Phase-58 *multi-round delayed-causal-evidence* regime,
  the *new harder regime* SDK v3.12 introduces. Two-round
  coordination where round-1 carries generic-noise-only mentions
  of (gold_A, gold_B, decoy) with the decoy cross-role-corroborated,
  AND round-2 carries one specific-tier disambiguating ``claim_kind``
  with NO ``service=`` token. **No single-round decoder can win on
  R-58 by construction (W11-Λ).** A *multi-round bundle decoder*
  that decodes the union of round-1 and round-2 admitted handoffs
  AND drops noise-corroborated decoys closes the gap (W11-1).
  Anchor:
  `vision_mvp/experiments/phase58_multi_round_decoder.py`.
* **R-60** — Phase-60 *open-world normalisation + real-Ollama
  transfer* regime, the *new harder regime* SDK v3.14 introduces.
  Two pre-committed sub-banks plus an opt-in real-LLM extension:
    - **R-60-wide** (synthetic_wide_oov_llm). Same R-58 delayed-
      causal-evidence shape but the LLM-shaped extractor emits kind
      variants from :data:`HEURISTIC_RESCUABLE_OOV_KINDS` —
      ``DEADLOCK_PROBABLY_DETECTED_MAYBE``, ``POOL_LOOKING_BUSY``,
      ``QUERY_SOMEWHAT_SLUGGISH``, ``DISK_GETTING_FULL_PROBABLY``,
      ``POOL_AT_CAPACITY``, ``P95_REGRESSION``, ``SLO_VIOLATED``,
      ``FW_DENIAL_BURST``, … — *outside* :data:`CLAIM_KIND_SYNONYMS`
      but *inside* the W13 heuristic abstraction closure. **No fixed-
      table normaliser can win on R-60-wide by construction**
      (W13-Λ-fixed at the open-world axis): every drifted variant
      survives :data:`CLAIM_KIND_SYNONYMS` unchanged and the priority
      decoder cannot match it. A *layered open-world normaliser*
      (``LayeredRobustMultiRoundBundleDecoder``: exact synonym table
      → heuristic abstraction rules → optional abstention, ahead of
      the W11 multi-round decoder) closes the gap (W13-1).
    - **R-60-cosmic** (synthetic_cosmic_oov_llm). Round-2 specific-
      tier disambiguating claims are replaced by truly arbitrary
      OOV tokens (``XYZZY_QQQQ``, ``COSMIC_RAY_FLIP``,
      ``INCIDENT_NUMBER_42``, …) — *outside both layers*. Both W12
      and W13 tie FIFO at 0.000; the *named open-world closure
      boundary* of any predicate-based normaliser (W13-4).
    - **R-60-ollama** (opt-in real-LLM extension; W13-Λ-real probe).
      Same R-58 events under the real Ollama Mac-1 producer (default
      ``qwen2.5:14b-32k``, optionally ``qwen3.5:35b``). The driver
      records every raw producer string into the report. The
      milestone *measures*, not *claims*, real-LLM transfer — see
      § 1.4 for the honest 4-tier grading on R-60-ollama.
  Anchor:
  `vision_mvp/experiments/phase60_open_world_normalization.py`.

* **R-59** — Phase-59 *real-LLM-driven multi-round delayed-
  disambiguation* regime, the *new harder regime* SDK v3.13
  introduces. Same two-round structural shape as R-58 but the
  candidate stream is generated by an **LLM-shaped extractor**
  (``synthetic_noisy_llm`` default, ``ollama`` opt-in) that emits
  *drifted* ``claim_kind`` strings (synonyms of the canonical kinds:
  ``DEADLOCK_DETECTED`` for ``DEADLOCK_SUSPECTED``,
  ``POOL_EXHAUSTED`` for ``POOL_EXHAUSTION``, …) and *drifted*
  ``service=`` payload tokens (``svc=X`` / ``for service X`` /
  ``service:X``). **No un-normalised single-round / multi-round
  decoder can win on R-59 by construction (W12-Λ at the real-LLM
  axis):** the synonym drift breaks ``_DECODER_PRIORITY`` lookups so
  the elected root_cause stays generic; the W11 contradiction-aware
  drop never fires. A *normalising multi-round bundle decoder*
  (``RobustMultiRoundBundleDecoder``: closed-vocabulary
  :data:`CLAIM_KIND_SYNONYMS` + payload-rewrite layer ahead of W11)
  closes the gap (W12-1). Anchor:
  `vision_mvp/experiments/phase59_real_llm_multi_round.py`.

## 1. Three pre-committed bars

> **SDK v3.14 anchors the bar to R-60.** Each version of this
> document anchors the *strict-gain regime* to the milestone-
> specific harder regime. SDK v3.9 anchored to R-55; SDK v3.10
> anchored to R-56; SDK v3.11 anchored to **R-57** AND introduced
> bar 7 (admission/decoding split). SDK v3.12 anchored to R-58 AND
> introduced bar 8 (temporal/structural split). SDK v3.13 anchored
> to R-59 AND introduced bar 9 (synthetic→real-LLM transfer split
> under bounded *fixed-vocabulary* producer noise). SDK v3.14
> anchors the bar to **R-60** below AND introduces **bar 10**
> (open-world normalisation split: a method must survive a producer-
> noise channel that *exceeds* the fixed-vocabulary closure when the
> drift remains inside the heuristic abstraction closure). Earlier
> R-55..R-59-anchored bars remain valid as historical bars; the
> *current* bar is R-60-anchored.

### 1.1 Strong success bar (a "real" advance)

A milestone *strongly advances* the thesis iff **all ten** hold (bars 1–6 always; bar 7 from SDK v3.11; bar 8 from SDK v3.12; bar 9 from SDK v3.13; bar 10 from SDK v3.14):

1. **Code anchor.** A new admission/decoder/coordination method
   ships in `vision_mvp/wevra/team_coord.py` (or sibling SDK
   module), exported from the SDK's public surface
   (`vision_mvp.wevra.__init__`), with a docstring that names its
   theorem family and falsifier.
2. **Strict gain on the named harder regime.** The new method
   strictly improves `accuracy_full` over substrate FIFO **and**
   over the previous SDK's strongest method (e.g. SDK v3.10
   `MultiServiceCorroborationAdmissionPolicy`, W9) on the
   *milestone-anchored harder regime* (currently **R-57** for
   SDK v3.11), by `≥ 0.20` pooled, on `n_eval ≥ 10` scenarios.
3. **Cross-bank stability.** The same gap `≥ 0.20` holds across
   `≥ 3` distinct `bank_seed` values, with no parameter retuning.
   Pre-commit the seed set in test code before running.
4. **No regression elsewhere.** On **R-53**, **R-54**, **R-55**,
   **and R-56**, the new method does not regress `accuracy_full`
   by more than `0.05` pooled relative to the SDK v3.10 baseline.
5. **Audit + lifecycle preserved.** Every capsule strategy passes
   the team-lifecycle audit (T-1..T-7) on every cell of every
   regime. `audit_ok_grid[strategy] = True` for all capsule
   strategies on R-53 / R-54 / R-55 / R-56 / R-57.
6. **Named bench property + falsifier regime.** The harder regime's
   structural property is named in code (mechanically verified by a
   test in `test_wevra_*.py`) and at least **one** *falsifier regime*
   is explicitly identified — a regime where the bench property does
   not hold and the new method does *not* beat FIFO (i.e. the
   conditionality is sharp).
7. **(SDK v3.11+) Admission/decoding split.** The milestone-
   anchored harder regime (R-57) is **provably insufficient for any
   service-blind admission policy** (W10-Λ holds: every admission-only
   policy in the SDK ties FIFO at 0.000 on R-57). The new method
   either (a) modifies the decoder, or (b) jointly modifies admission
   AND decoder. A pure new admission policy does NOT clear bar 7 on
   SDK v3.11+.
8. **(SDK v3.12 only) Temporal/structural split.** The milestone-
   anchored harder regime (R-58) is **provably insufficient for any
   single-round decoder** (W11-Λ holds: every single-round capsule
   strategy — including the SDK v3.11 W10 bundle decoder — ties
   FIFO at 0.000 on R-58 default). The new method must cross the
   cross-round boundary (i.e. consume admitted handoffs from ≥ 2
   rounds and produce a single decoded answer). A pure
   single-round method does NOT clear bar 8 on SDK v3.12.
10. **(SDK v3.14+) Open-world normalisation split.** The milestone-
   anchored harder regime (R-60-wide) is **provably insufficient for
   any fixed-closed-vocabulary normaliser** (W13-Λ-fixed at the
   open-world axis: every entry in
   :data:`HEURISTIC_RESCUABLE_OOV_KINDS` is *not* in
   :data:`CLAIM_KIND_SYNONYMS`, so :class:`RobustMultiRoundBundleDecoder`
   ties FIFO at 0.000 on R-60-wide default). The new method must
   include an explicit *closure-widening* layer beyond exact lookup —
   either heuristic regex predicates over the kind text, embedding-
   distance lookups, or a learned normaliser. A pure exact-table
   method does NOT clear bar 10 on SDK v3.14. **Pre-committed bench
   property:** the layered normaliser's heuristic rules must resolve
   *every* variant in :data:`HEURISTIC_RESCUABLE_OOV_KINDS` to its
   named canonical kind (mechanically verified by
   ``W13ClosureTests::test_every_wide_oov_variant_outside_w12_inside_w13``).
   **Pre-committed falsifier:** when the producer LLM emits a *truly
   arbitrary* kind outside both layers (R-60-cosmic regime, every
   entry in :data:`COSMIC_OOV_KINDS`), the layered normaliser ties
   FIFO at 0.000 (W13-4). The closure boundary is structural for any
   predicate-based normaliser; widening it is a research move
   (W13-C2 / W13-C3), not a structural fix.

9. **(SDK v3.13 only) Real-LLM transfer / bounded-producer-noise
   split.** The milestone-anchored harder regime (R-59) is
   **provably insufficient for any un-normalised single-round /
   multi-round decoder** under the named bounded-producer-noise
   channel (W12-Λ holds at the real-LLM axis: every un-normalised
   capsule strategy — including the SDK v3.12 W11 multi-round bundle
   decoder — ties FIFO at 0.000 on R-59 default at
   ``synonym_prob=0.50, svc_token_alt_prob=0.30``). The new method
   must include an explicit *normalisation layer* (closed-vocabulary
   :data:`CLAIM_KIND_SYNONYMS` + payload rewrite) that maps the
   producer-noise channel into the canonical kind / payload
   vocabulary the W11 / W10 decoder consumes. A pure synthetic
   method that does not run the normalisation layer does NOT clear
   bar 9 on SDK v3.13. **Pre-committed bench property:** the
   ``CLAIM_KIND_SYNONYMS`` table must contain every variant the
   noisy LLM extractor can emit (``NOISY_KIND_VARIANTS``) — this is
   the closure property mechanically verified by
   ``NoisyExtractorTests::test_noisy_variants_all_in_synonym_table``.
   **Pre-committed falsifier:** when the producer LLM emits an
   *out-of-vocabulary* kind (named in
   ``OUT_OF_VOCAB_KINDS`` and not in ``CLAIM_KIND_SYNONYMS``),
   normalisation cannot rescue it and the W12 method ties FIFO at
   0.000 (W12-4). The ``ollama`` opt-in mode is the *honest extension
   path*: when Mac 1 returns a real LLM stream that satisfies the
   bench property, the same R-59 strong bar applies; if the LLM's
   drift falls outside the synonym closure the bar is *not* met
   and the milestone is partial — see § 1.2.

If a milestone clears 1.1, the canonical phrasing is:

> *"On R-`<N>` (named bench property), method M strictly improves
> accuracy_full by ≥ 0.20 over both substrate FIFO and the previous
> SDK's strongest method (named), stable across ≥ 3 bank seeds, with
> no regression on R-`<{others}>` and audit_ok preserved on every
> cell. The win does not transfer to the falsifier regime F (named)."*

For SDK v3.10 specifically, the canonical phrasing instantiates as:

> *"On R-56 (multi-service-gold + cross-role-corroborated), the
> `MultiServiceCorroborationAdmissionPolicy` (W9) strictly improves
> accuracy_full by +1.000 over both substrate FIFO and the SDK v3.9
> `CrossRoleCorroborationAdmissionPolicy` (W8), stable across 5/5
> bank seeds, with no regression on R-53 / R-54 / R-55 and audit_ok
> preserved on every cell. The win does not transfer to the
> Phase-56-falsifier regime where the decoy is also corroborated by
> ≥ 2 distinct producer roles (W9-4)."*

For SDK v3.11 specifically, the canonical phrasing instantiates as:

> *"On R-57 (multi-service-gold + corroborated-decoy), no
> service-blind admission policy in the SDK (FIFO, priority,
> coverage, W7-2 cohort, W8 corroboration, W9 multi-service)
> improves accuracy_full above 0.000 — this is the structural
> admission-only limitation (W10-Λ). The new
> `BundleAwareTeamDecoder` (W10) — paired with the W9 admission
> policy as the joint admit-and-decode method (W10-AD) — strictly
> improves accuracy_full by ≥ 0.20 over both substrate FIFO and the
> SDK v3.10 W9 admission, stable across ≥ 3 bank seeds, with no
> regression on R-53 / R-54 / R-55 / R-56 and audit T-1..T-7
> preserved on every cell. The win does not transfer to the
> Phase-57-falsifier regime where the decoy is corroborated AND
> mentioned via a causal claim_kind for the chosen root_cause
> (W10-4)."*

For SDK v3.14 specifically, the canonical phrasing instantiates as:

> *"On R-60-wide (open-world drift, every variant in
> :data:`HEURISTIC_RESCUABLE_OOV_KINDS` outside the W12 fixed table),
> no fixed-closed-vocabulary normaliser in the SDK improves
> accuracy_full above 0.000 — this is the W13-Λ-fixed limit. The new
> ``LayeredRobustMultiRoundBundleDecoder`` (W13) — exact synonym
> table → heuristic abstraction rules → optional abstention, ahead
> of the W11 multi-round decoder — achieves accuracy_full = 1.000,
> strictly improving over both substrate FIFO (+1.000) and the SDK
> v3.13 W12 ``RobustMultiRoundBundleDecoder`` (+1.000), stable across
> 5/5 alternate (bank_seed, llm_seed) values with min gap +0.917
> well above the strong-bar 0.50 threshold, with no regression on
> R-53 / R-54 / R-55 / R-56 / R-57 / R-58 / R-59 / R-60-clean and
> audit T-1..T-7 preserved on every cell. The win does not transfer
> to the R-60-cosmic falsifier where the producer emits truly
> arbitrary OOV (W13-4)."*

Honest R-60-ollama partial-success reading (when the milestone runs
the opt-in real-Ollama mode):

> *"On R-60-ollama (real Mac-1 ``qwen2.5:14b-32k`` on the Phase-58
> events), the R-60 strong bar is **NOT** cleared. The 14B model
> emits *canonical* claim kinds (zero kind drift across n=4 ×
> 12 producer calls) and *filters out low-magnitude decoy events*
> as noise (zero cross-role decoy corroboration in 4/4 scenarios).
> The bench property does not hold in 0/4 scenarios. Normalisation
> (W12 / W13) has nothing to rescue. The honest empirical reading is
> W13-Λ-real: real Ollama 14B at temperature 0 on the calibrated
> incident-triage prompt does not, by default, generate the R-58
> delayed-causal-evidence shape that the synthetic noisy extractor
> mimics. The synthetic→real-LLM transfer story therefore has a fifth
> layer: prompt-side discipline + event-shape design (not just
> normalisation) is required to elicit the bench property from a real
> producer. R-60-ollama is the *measurement* anchor; the W13 advance
> is on R-60-wide synthetic, where the W13 method does materially
> better than W12."*

For SDK v3.13 specifically, the canonical phrasing instantiates as:

> *"On R-59 (real-LLM-driven multi-round delayed-disambiguation,
> ``synonym_prob=0.50, svc_token_alt_prob=0.30`` synthetic-noisy-LLM
> extractor), no un-normalised single-round / multi-round capsule
> strategy in the SDK (FIFO, priority, coverage, W7-2 cohort, W8
> corroboration, W9 multi-service, W10 bundle decoder, **and SDK
> v3.12 W11 ``MultiRoundBundleDecoder``**) improves accuracy_full
> above 0.000 — this is the structural real-LLM-axis limitation
> (W12-Λ at the real-LLM axis). The new
> ``RobustMultiRoundBundleDecoder`` (W12) — closed-vocabulary
> :data:`CLAIM_KIND_SYNONYMS` + payload-rewrite layer ahead of the
> W11 multi-round decoder — achieves accuracy_full = 1.000, strictly
> improving over both substrate FIFO (+1.000) and the SDK v3.12 W11
> unmodified decoder (+0.750..+1.000 per seed), stable across 5/5
> alternate (bank_seed, llm_seed) values, with no regression on
> R-53 / R-54 / R-55 / R-56 / R-57 / R-58 / R-59 clean and audit
> T-1..T-7 preserved on every cell. The win does not transfer to
> the Phase-59-falsifier regime where the LLM emits *out-of-
> vocabulary* kinds outside ``CLAIM_KIND_SYNONYMS`` (W12-4)."*

### 1.2 Partial success bar (a real but narrower advance)

A milestone *partially advances* the thesis iff **all four** hold:

1. **Code anchor** (same as 1.1).
2. **Either-or gain.** Either:
   * a strict gain `≥ 0.20` on the milestone-anchored harder regime
     (R-56 for SDK v3.10) but a regression `> 0.05` on R-53 / R-54 /
     R-55 (a *trade-off* result), OR
   * a strict gain `≥ 0.10` on **three** of {R-53, R-54, R-55, R-56}
     with no regression `> 0.05` on the fourth.
3. **Audit preserved.** As in 1.1.5.
4. **One named theorem-empirical claim** with stated falsifier.

The canonical phrasing for partial success names the trade-off
explicitly: *"...wins R-55 by +X but trades R-54 by −Y; net
mean across the three regimes is +Z..."*. No "broad win" framing.

### 1.4 R-60-ollama 4-tier grading (SDK v3.14 only)

R-60-ollama is the **first real-LLM probe in the programme that runs
through the full team-coord pipeline**. It is graded *separately* from
R-60-wide because real-LLM behaviour is empirically observed (not
designed). The grading has four pre-committed tiers:

* **R-60-OLLAMA-A (strong real transfer).** The bench property holds
  in ≥ 75 % of scenarios under the real LLM, the LLM emits drifted
  kinds in ≥ 20 % of causal claims (≥ 50 % of those *outside*
  :data:`CLAIM_KIND_SYNONYMS` so W12 cannot rescue), AND the W13
  layered method strictly improves over the W12 method by ≥ 0.20
  pooled accuracy_full. Anchor: ``run_phase60(llm_mode='ollama',
  n_eval=12, model='qwen2.5:14b-32k', endpoint=...)``.
* **R-60-OLLAMA-B (partial real transfer).** The bench property
  holds in ≥ 50 % of scenarios AND either (i) the LLM emits enough
  W13-rescuable drift that the W13 advance is non-zero pooled, OR
  (ii) the LLM emits W12-rescuable drift and W12 already wins
  (in which case W13 ties W12 — a backward-compat win, not a strict
  W13 advance).
* **R-60-OLLAMA-C (null real transfer; honest negative).** The
  bench property holds in < 50 % of scenarios AND/OR the LLM emits
  zero drift on causal claims. Normalisation (W12 / W13) has
  nothing to rescue. The synthetic→real-LLM transfer story is
  dominated by event-shape mismatch / prompt-side discipline rather
  than by normalisation.
* **R-60-OLLAMA-D (failure).** Audit breaks on any cell, OR the LLM
  endpoint is unreachable / fails on > 50 % of producer calls.

R-60-OLLAMA-A clears bar 10 *and* clears the historical bar 9
(W12-C2 — real-Ollama transfer of W12-1) end-to-end and is the
strongest possible empirical anchor.

R-60-OLLAMA-C is the **honest current reading** for SDK v3.14:
real Ollama ``qwen2.5:14b-32k`` on the Phase-58 events at
temperature 0 emits canonical kinds and filters low-magnitude
decoy events. The bench property does not hold in any scenario;
the W13 advance is structurally invisible because there is no
drift for either layer to rescue. SDK v3.14 documents this as a
**partial success** at the milestone level (the W13 wide-OOV win
is the strong-success anchor; R-60-ollama is the partial-real
anchor). See `docs/RESULTS_WEVRA_OPEN_WORLD_NORMALIZATION.md`
§ 4 and § 6 for the honest scope statement.

### 1.3 Falsifying failure bar (the milestone is null)

A milestone is a *falsifying failure* iff **any** holds:

1. The new method does not strictly improve any single regime by
   `≥ 0.10` and the cross-regime mean gain is `≤ 0.0`.
2. The new method introduces a regression `> 0.10` on R-53, R-54,
   or R-55 *and* the gain on the milestone-anchored harder regime
   (R-56 for SDK v3.10) does not exceed the regression magnitude.
3. The lifecycle audit breaks on any cell.
4. The W7-2 contract test
   (`test_buffered_cohort_strictly_beats_fifo_at_K4`) fails.
5. The W8-1 contract test
   (`test_corroboration_strictly_beats_W7_2_at_K4`) fails.

A null milestone is *not* a sin. The sin is calling a null
milestone an advance.

## 2. Required ingredients of R-55 (SDK v3.9)

The "harder fair regime" introduced by SDK v3.9 must combine:

* **Cross-role complementary evidence.** Each scenario's gold answer
  requires aggregating signals from `≥ 3` distinct producer roles.
  No single role is sufficient.
* **Budget pressure.** `|candidates_to_auditor| > K_auditor` on
  every scenario (mechanically tested).
* **Locally-misleading distractors.** For each role, the role's
  local plurality of `service=<tag>` mentions is a **decoy** — i.e.
  if the auditor naively picked the role-local plurality, it would
  pick the wrong service. The cross-role aggregate must point to
  the gold service.
* **Foreign-service decoys + decoy raw plurality.** The auditor
  sees `≥ 2` distinct decoy services. **Some decoy service must
  have strictly more raw mentions than the gold service** — so
  W7-2 buffered cohort (single-tag plurality) picks the *decoy*
  and ties FIFO at 0.000. This is the **strict separation** from
  the W7-2 win.
* **Cross-role corroboration of gold.** The gold service is
  mentioned by **strictly more distinct producer roles** than any
  decoy service. This is the signal the new method must be able to
  exploit.
* **Realistic claim_kinds.** The gold claims carry the *causal*
  claim_kinds (those that the priority decoder maps to the right
  root_cause); decoy claims carry plausible-but-not-causal
  claim_kinds.

A regime missing any of these is *not* R-55 — it does not test
the SDK v3.9 hypothesis.

## 2.5. Required ingredients of R-56 (SDK v3.10)

The "harder fair regime" introduced by SDK v3.10 (the *current*
strict-gain anchor) must combine:

* **Multi-service gold answer.** `gold_services` of size **≥ 2**.
  Both gold services must appear in the auditor's admitted set for
  `services_correct` to hold (set equality). This is what makes W8
  *single-tag* corroboration insufficient: it admits only the top-1
  corroborated tag.
* **Both gold services cross-role corroborated.** Each gold service
  is mentioned by **≥ 2 distinct producer roles** in the candidate
  stream (mechanically tested by
  `Phase56BankShapeTests.test_default_bank_multi_service_gold_property_holds`).
* **Single-role decoy storm with raw plurality.** At least one
  decoy service has raw plurality (more total mentions than each
  gold service) AND is corroborated by exactly 1 producer role.
  This makes substrate FIFO and W7-2 buffered cohort (single-tag
  plurality) fail: they admit the decoy.
* **Budget pressure.** `|candidates_to_auditor| > K_auditor` on
  every scenario.
* **Realistic claim_kinds.** The decoder priority over admitted
  claim_kinds maps to the gold `root_cause` (no claim-kind
  override).
* **Audit-preserving by construction.** The team-lifecycle audit
  (T-1..T-7) holds on every cell of every regime (mechanically
  tested by `W9LifecycleAuditTests`).

A regime missing any of these is *not* R-56 — it does not test
the SDK v3.10 hypothesis.

## 2.6. Required ingredients of R-57 (SDK v3.11)

The "harder fair regime" introduced by SDK v3.11 — the *current*
strict-gain anchor, and the regime where admission alone is
provably insufficient — must combine:

* **Multi-service gold answer.** `gold_services` of size **≥ 2**
  (Phase-57 default uses size 2). Set-equality grading, same as R-56.
* **Both gold services cross-role corroborated by *causal*
  claim_kinds.** For the scenario's `gold_root_cause`, define
  `CCK(gold_root_cause)` as the set of claim_kinds the
  `_decoder_from_handoffs` priority table maps to the same root_cause
  label OR to a same-tier root_cause (data-tier / storage-tier /
  network-tier / generic). Each gold service must be mentioned by
  ≥ 2 distinct producer roles via at least one claim_kind in CCK.
* **Decoy corroborated by ≥ `min_corroborated_roles` distinct roles
  via *non-causal* claim_kinds.** The decoy_storm service must be
  mentioned by ≥ 2 distinct producer roles via claim_kinds that are
  NOT in `CCK(gold_root_cause)` (typically generic noise:
  `LATENCY_SPIKE`, `ERROR_RATE_SPIKE`, `FW_BLOCK_SURGE`). This is
  the *structural separator* that makes the bundle-aware decoder
  necessary: admission cannot distinguish the corroborated decoy
  from corroborated gold, but the decoder can — once the
  high-priority gold-causal claim is admitted, the decoder reads
  off `root_cause`, then filters services by CCK.
* **Decoy raw plurality in W9 score.** The decoy's raw mention
  count is high enough that with `top_k=2` the decoy beats at least
  one gold service in the W9 dominant set (mechanically tested by
  `Phase57BankShapeTests.test_W9_admission_misses_one_gold`).
* **Budget pressure.** `|candidates_to_auditor| > K_auditor` on
  every scenario.
* **Audit-preserving by construction.** The team-lifecycle audit
  (T-1..T-7) holds on every cell of every regime (mechanically
  tested by `W10LifecycleAuditTests`).

A regime missing any of these is *not* R-57 — it does not test
the SDK v3.11 hypothesis.

## 2.7. Required ingredients of R-58 (SDK v3.12)

The "harder fair regime" introduced by SDK v3.12 must combine:

* **Two coordination rounds.** Both rounds run through the same
  ``TeamCoordinator`` and seal a per-round ``ROLE_VIEW`` capsule.
* **Round-1 generic-noise-only on the auditor stream.** Every
  round-1 emission to the auditor has ``claim_kind`` in the
  closed-vocabulary generic-noise set
  ``{LATENCY_SPIKE, ERROR_RATE_SPIKE, FW_BLOCK_SURGE}``. The
  priority decoder elects a generic root_cause on round-1 alone.
* **Round-2 specific-tier with no service= tokens.** The round-2
  emission carries a single specific-tier kind (DEADLOCK_SUSPECTED,
  POOL_EXHAUSTION, DISK_FILL_CRITICAL, …) and *no*
  ``service=<tag>`` token. Round-2 alone has empty service set.
* **Decoy-only-in-round-1.** The decoy service is mentioned only
  in round 1.
* **Round-1 decoy is cross-role corroborated.** ≥ 2 distinct
  producer roles mention the decoy via generic-noise kinds.
* **Audit-preserving.** T-1..T-7 holds on every cell of every
  capsule strategy.

A regime missing any of these is *not* R-58 — it does not test
the SDK v3.12 hypothesis.

## 2.8. Required ingredients of R-59 (SDK v3.13)

The "harder fair regime" introduced by SDK v3.13 — the *current*
strict-gain anchor and the regime where un-normalised cross-round
decoders are provably insufficient — must combine:

* **All R-58 structural ingredients.** R-59 retains the two-round
  delayed-causal-evidence shape from R-58 unchanged. The W11
  contradiction-aware drop is the *target* the W12 normalisation
  layer has to be able to feed.
* **An LLM-shaped extractor between the events and the candidate
  stream.** The default mode (``synthetic_noisy_llm``) uses a
  deterministic in-process extractor that mimics realistic LLM
  drift; the opt-in mode (``ollama``) uses the same Phase-53-style
  prompt against a real LLM endpoint. Both feed into the same
  TeamCoordinator pipeline.
* **Bounded claim-kind drift channel** (``synonym_prob ≥ 0.30``
  by default at 0.50). Each canonical kind has ≥ 1 plausible LLM
  variant in ``NOISY_KIND_VARIANTS`` and *every* such variant is
  in ``CLAIM_KIND_SYNONYMS`` (closed-vocabulary closure ).
* **Bounded payload drift channel** (``svc_token_alt_prob ≥ 0.20``
  by default at 0.30). The LLM may emit alternative spellings of
  ``service=<tag>`` (``svc=X`` / ``for service X`` / ``service:X``);
  every such spelling matches a pattern in
  :data:`_SERVICE_TAG_REWRITES`.
* **Out-of-vocabulary kinds appear only in the falsifier bank.**
  The default bank's drift is bounded by the synonym closure; the
  falsifier bank (``oov_prob = 0.50`` set in ``run_phase59`` when
  ``use_falsifier_bank=True``) emits novel kinds outside the
  closure.
* **Determinism.** The synthetic noisy extractor is RNG-deterministic
  given ``(bank_seed, llm_seed, scenario_id, round_idx)``. The
  Ollama mode is best-effort deterministic via ``temperature=0``;
  HTTP failures degrade to the synthetic fallback (which is
  *labelled* in the report's ``extractor_stats`` block).
* **Audit-preserving.** T-1..T-7 holds on every cell of every
  capsule strategy on every (LLM mode, bank_seed) cell.

A regime missing any of these is *not* R-59 — it does not test
the SDK v3.13 hypothesis.

## 2.9. Required ingredients of R-60 (SDK v3.14)

The "harder fair regime" introduced by SDK v3.14 — the *current*
strict-gain anchor and the regime where fixed-vocabulary normalisers
are provably insufficient — must combine:

* **All R-58 structural ingredients.** R-60 retains the two-round
  delayed-causal-evidence shape from R-58 unchanged. The W11
  contradiction-aware drop is the *target* the W13 normalisation
  layer has to be able to feed, just like W12.
* **An LLM-shaped extractor between the events and the candidate
  stream.** Phase-60 ships three modes: ``synthetic_clean_llm``,
  ``synthetic_wide_oov_llm`` (the W13-1 anchor), and
  ``synthetic_cosmic_oov_llm`` (the W13-4 anchor). The opt-in
  ``ollama`` mode is the R-60-ollama probe — § 1.4.
* **Drift channel that exceeds the W12 fixed-vocabulary closure.**
  The R-60-wide bank emits variants from
  :data:`HEURISTIC_RESCUABLE_OOV_KINDS` whose *every* entry is
  *not* in :data:`CLAIM_KIND_SYNONYMS`. Mechanically verified by
  ``W13ClosureTests::test_every_wide_oov_variant_outside_w12_inside_w13``.
* **Drift channel that stays inside the W13 heuristic closure.**
  The R-60-wide bank's variants must *all* resolve to their named
  canonical kinds via :class:`LayeredClaimNormalizer` (the
  closure-widening contract). Mechanically verified by the same
  test.
* **Cosmic-OOV falsifier bank (R-60-cosmic).** The Phase-60
  ``synthetic_cosmic_oov_llm`` extractor replaces every round-2
  specific-tier disambiguating claim with an entry from
  :data:`COSMIC_OOV_KINDS`; *every* entry is verified absent from
  :data:`CLAIM_KIND_SYNONYMS` AND verified *not* to resolve through
  the heuristic layer. Mechanically verified by
  ``W13ClosureTests::test_every_cosmic_oov_variant_escapes_both_layers``.
* **Determinism.** The Phase-60 synthetic extractor is RNG-
  deterministic given ``(bank_seed, llm_seed, scenario_id,
  round_idx)``.
* **Audit-preserving.** T-1..T-7 holds on every cell of every
  capsule strategy on every (LLM mode, bank_seed) cell.

A regime missing any of these is *not* R-60 — it does not test
the SDK v3.14 hypothesis.

## 3. What we are explicitly NOT testing

* **Not** "does cohort coherence ever beat FIFO?" — that's W7-2,
  already shipped and conditional.
* **Not** "does the learned policy generalise OOD?" — the W6-C2
  falsification in SDK v3.7 closed that question; learning is not
  the right tool here.
* **Not** "does scaling the LLM solve coordination?" — W6-C1
  closed that.
* **Not** "is the runtime fully capsule-native?" — see
  `HOW_NOT_TO_OVERSTATE.md`.

## 4. How to use this document

* Before a milestone starts: declare which bar (1.1 / 1.2 / 1.3)
  the milestone is targeting. If no bar is declared, the default
  is 1.1.
* During the milestone: do not relax the bar after seeing partial
  results. If the milestone ends up at 1.2 or 1.3, label it
  honestly.
* After the milestone: the milestone note must cite this document
  and state which bar was cleared. Reviewers reject milestone
  notes that overstate.

## Cross-references

* Theorem registry: `docs/THEOREM_REGISTRY.md`
* Research status: `docs/RESEARCH_STATUS.md`
* Overstatement guard: `docs/HOW_NOT_TO_OVERSTATE.md`
* Master plan: `docs/context_zero_master_plan.md` § 4.26 (SDK v3.9)
