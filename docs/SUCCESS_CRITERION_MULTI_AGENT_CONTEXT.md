# Success criterion — solving multi-agent context (SDK v3.17 bar)

> Pre-committed, falsifiable bar for what counts as a *real* advance
> on "solving multi-agent context" in the Context Zero / Wevra
> programme. This document is the **referee** for SDK v3.9 / v3.10 /
> v3.11 / v3.12 / v3.13 / v3.14 / v3.15 / v3.16 / v3.17 (and later
> milestones). Any milestone note that claims an advance must name
> the bar it cleared and cite the code-anchored evidence. Last
> touched: SDK v3.17, 2026-04-27.
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

* **R-63** — Phase-63 *end-to-end W14 + W15 composition* regime,
  the *new harder regime* SDK v3.17 introduces. R-63 is the first
  programme regime where **both** the producer-side ambiguity-
  preservation layer (W14) AND the decoder-side capsule context-
  packing layer (W15) are independently load-bearing on the *same*
  cell. The events are R-62's multi-hypothesis comparable-magnitude
  shape; the producer is Phase-61's deterministic
  ``MagnitudeFilteringExtractor`` with its ``prompt_mode`` toggle
  (naive collapses round-1 toward top-N events per (role, kind);
  structured preserves the per-event mandate); the decoder
  consumes the cross-round bundle under a strict ``T_decoder``.
  Six pre-committed sub-banks plus a falsifier and three
  ollama-replay cells:
    - **R-63-baseline-loose** (identity producer, ``T_decoder=None``).
      Sanity anchor: cross-round capsule decoders saturate at 1.000.
      W16-3 backward-compat anchor on R-62 events.
    - **R-63-W15-only** (identity, ``T_decoder=24``). Identical to
      R-62-tightbudget; isolates the W15 contribution.
    - **R-63-W14-only** (mag-filter naive, ``T_decoder=None``).
      Producer-side compression breaks the bench property even on
      R-62 multi-hypothesis events; every cross-round decoder ties
      FIFO at 0.000. W14-Λ-prompt extension to R-62 events.
    - **R-63-naive-tight** (mag-filter naive, ``T_decoder=24``).
      Joint failure of W14-Λ-prompt and W15-Λ-budget on the same
      regime; every capsule strategy ties FIFO at 0.000.
      **W16-Λ-compose anchor.**
    - **R-63-W14-success** (mag-filter structured, ``T_decoder=None``).
      W14 alone restores the bench property; cross-round wins return
      to 1.000. The composed packer ties W13 byte-for-byte.
    - **R-63-COMPOSED-TIGHT** (mag-filter structured,
      ``T_decoder=24``). **The W16-1 anchor.** W14 restores the
      bench property AND the decoder budget bites simultaneously;
      only the AttentionAwareBundleDecoder over the structured-
      prompt stream wins at 1.000; FIFO-packed-W13 (the
      W14-only-budgeted baseline) collapses to 0.000;
      AttentionAwareBundleDecoder over the *naive*-prompt stream
      (the W15-only-without-W14 baseline) also collapses to 0.000.
      **+1.000 strict separation** vs the strongest non-composed
      baseline, stable across 5/5 alternate ``bank_seed`` values.
    - **R-63-degen-budget** (mag-filter structured,
      ``T_decoder=2``). **W16-Λ-degenerate falsifier** — the budget
      is below the round-2 specific claim's token cost; both
      packers collapse to 0.000.
    - **R-63-OLLAMA-REPLAY-LOOSE / -COMPOSED-TIGHT / -NAIVE-TIGHT**
      (replay extractor over recorded Phase-61
      ``qwen2.5:14b-32k`` bytes; ``K_auditor = 8``,
      ``T_decoder ∈ {None, 14}``). **W16-Λ-real-replay anchor.**
      Replay-loose reproduces the Phase-61 W14-Λ-real anchor at
      0.500. Replay-composed-tight at ``T_decoder = 14`` shows
      ``capsule_attention_aware = 0.500`` and
      ``capsule_layered_fifo_packed = 0.000`` — **+0.500 strict
      gain** over the FIFO-packed-W14-only baseline on a *real-LLM
      stream*. Replay-naive-tight reproduces the joint-failure
      regime on recorded LLM bytes.
  Anchor:
  `vision_mvp/experiments/phase63_composed_real_llm.py`.

* **R-62** — Phase-62 *attention-aware capsule context packing under
  decoder-side budget pressure* regime, the *new harder regime* SDK
  v3.16 introduces. The producer-side gap was closed by W14 on R-61;
  R-62 attacks the symmetric *downstream* gap: even when the bench
  property holds and the cross-round capsule pipeline is in scope, a
  strict *decoder-side* token budget ``T_decoder`` can drop the
  round-2 specific-tier disambiguator under naive FIFO packing of the
  admitted union. R-62 attacks this directly via two paired moves:
    - **Multi-hypothesis comparable-magnitude events.** Each scenario
      carries 2 gold services + 2 decoy services. Every decoy service
      is mentioned by ≥ 2 distinct producer roles in round 1 (the W11
      noise-decoy-drop trigger condition); every gold service is
      mentioned by exactly 1 distinct producer role in round 1 (so
      the W11 drop preserves golds). Multi-hypothesis means: 4
      service hypotheses survive naive admission; the disambiguation
      is recoverable *only* from the round-2 specific causal claim
      AND cross-role corroboration of round-1 noise.
    - **Decoder-side token budget.** A new strict budget ``T_decoder``
      is enforced over the cross-round bundle the decoder consumes.
      Default is loose (R-62-default sanity anchor); the
      R-62-tightbudget regime sets ``T_decoder`` strictly below the
      admitted union's token sum so the budget bites.
  Three pre-committed sub-banks plus a named falsifier:
    - **R-62-default** (synthetic, ``T_decoder=None``). Sanity
      anchor: every cross-round decoder (W11 / W12 / W13 / W15)
      saturates at ``accuracy_full = 1.000``. The W15 method ties
      W13 byte-for-byte (W15-3 backward-compat anchor).
    - **R-62-tightbudget** (synthetic, ``T_decoder=24``). The W15-1
      anchor. Under FIFO packing the round-2 specific claim falls
      past the budget and ``capsule_layered_fifo_packed`` ties FIFO
      at ``accuracy_full = 0.000``. The W15
      :class:`AttentionAwareBundleDecoder` reorders the union by
      salience (CCK + tier + corroboration + magnitude + round) and
      packs greedy with hypothesis preservation; ``accuracy_full =
      1.000``. The strict separation is +1.000 vs FIFO-packed-W13,
      stable across 5/5 alternate ``bank_seed`` values.
    - **R-62-saturation** (synthetic, ``T_decoder=9999``; W15-Λ-budget
      named falsifier). Under no budget pressure, salience reordering
      is a no-op on the answer field — both W15 and FIFO-pack hit
      1.000 alike. The W15-1 win is *conditional* on budget
      pressure; this regime makes the conditionality sharp.
  Anchor:
  `vision_mvp/experiments/phase62_attention_aware_packing.py`.

* **R-61** — Phase-61 *producer-side ambiguity-preservation* regime,
  the *new harder regime* SDK v3.15 introduces, anchored on the
  W13-Λ-real diagnosis. R-58..R-60's bottleneck under a real LLM is
  **not** kind drift but *upstream filtering*: the producer LLM
  silently discards low-magnitude decoy events as noise and
  compresses the round-1 stream toward a single best diagnosis. R-61
  attacks this directly via two paired moves:
    - **Comparable-magnitude events.** The decoy storm service is no
      longer a deliberately *low-magnitude* impostor: it is a real
      collateral-damage service whose round-1 spikes have the same
      operational magnitudes as the gold services (``p95_ms``,
      ``error_rate``, firewall-deny counts). The disambiguation is
      *only* recoverable from the round-2 specific causal claim, not
      from magnitude. Formal property in §§ 2.10. Mechanically
      verified by ``Phase61BankShapeTests``.
    - **Structured producer protocol.** An opt-in prompt mode
      (``StructuredProducerProtocol``) splits round 1 (operational
      *observation*: "list every distinct event you see — describe,
      do not diagnose") from round 2 (specific *diagnosis*: "what is
      the underlying cause?"), AND requires the LLM to emit one
      claim per listed event with no compressed summarisation.
      The protocol is the W14-1 candidate for closing the
      W13-Λ-real producer-filtering gap.
  Three pre-committed sub-banks plus an opt-in real-LLM extension:
    - **R-61-default** (synthetic-comparable-magnitude). Same
      bench-property witness checks as R-58 / R-60 (delayed-causal-
      evidence + cross-role decoy corroboration), but with
      magnitude-balanced decoy events. The W11/W12/W13 decoders all
      win at synthetic — R-61-default is the *baseline* / sanity
      anchor (W14-3 backward-compat target) showing the redesigned
      events do not break the synthetic property.
    - **R-61-naive-prompt** (synthetic, magnitude-filter simulation).
      A deterministic "magnitude-filter" extractor mimics the
      observed real-LLM behaviour on the *original* prompt:
      magnitude < threshold ⇒ emit NONE; round-1 best-claim
      compression ⇒ emit only the highest-magnitude observation.
      Bench property holds in 0/8 by construction; W11/W12/W13 tie
      FIFO. **W14-Λ-prompt:** any decoder is structurally
      insufficient when the producer collapses ambiguity upstream.
      This is the synthetic counterpart of W13-Λ-real.
    - **R-61-structured-prompt** (synthetic, structured-protocol
      simulation). Same magnitude-filter extractor wired to the
      structured protocol's prompt-side schema (every listed event
      ⇒ one claim, observation ≠ diagnosis). Bench property holds
      in N/8; the W11..W13 decoders' headroom returns. **W14-1
      candidate.**
    - **R-61-ollama** (opt-in real-LLM probe; W14-Λ-real anchor).
      Talks Mac-1 ``qwen2.5:14b-32k`` on the redesigned events under
      the structured protocol; raw producer responses captured into
      the report. Honest 4-tier grading mirrors § 1.4 — see § 1.5.
      The protocol's contribution is *measured*, not *claimed*: if
      the real LLM still filters or compresses, the structural
      diagnosis sharpens; if the real LLM preserves observation,
      W13-C3 advances toward discharge.
  Anchor:
  `vision_mvp/experiments/phase61_producer_ambiguity_preservation.py`.

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

> **SDK v3.17 anchors the bar to R-63.** Each version of this
> document anchors the *strict-gain regime* to the milestone-
> specific harder regime. SDK v3.9 anchored to R-55; SDK v3.10
> anchored to R-56; SDK v3.11 anchored to **R-57** AND introduced
> bar 7 (admission/decoding split). SDK v3.12 anchored to R-58 AND
> introduced bar 8 (temporal/structural split). SDK v3.13 anchored
> to R-59 AND introduced bar 9 (synthetic→real-LLM transfer split
> under bounded *fixed-vocabulary* producer noise). SDK v3.14
> anchored to **R-60** AND introduced bar 10 (open-world
> normalisation split: a method must survive a producer-noise
> channel that *exceeds* the fixed-vocabulary closure when the drift
> remains inside the heuristic abstraction closure). SDK v3.15
> anchored the bar to **R-61** AND introduced **bar 11**
> (producer-side ambiguity-preservation split: a method must clear
> the *prompt-side discipline* gate that W13-Λ-real identified as
> the dominant blocker on real-LLM transfer; a downstream-only
> method that does not address producer-side compression of the
> bench property does NOT clear bar 11). SDK v3.16 anchored the bar
> to **R-62** AND introduced **bar 12**
> (joint-correctness-and-context-efficiency split: a method must
> clear *both* an accuracy threshold AND an explicit decoder-side
> token-budget constraint; a method that wins on accuracy alone
> while ignoring decoder context budget does NOT clear bar 12).
> SDK v3.17 anchors the bar to **R-63** below AND introduces
> **bar 13** (end-to-end composition split: a method must clear
> a regime where *both* W14 producer-protocol ingredients AND W15
> decoder-budget ingredients are independently load-bearing; a
> method that closes one structural axis but leaves the other
> firing does NOT clear bar 13).
> Earlier R-55..R-62-anchored bars remain valid as historical
> bars; the *current* bar is R-63-anchored.

### 1.1 Strong success bar (a "real" advance)

A milestone *strongly advances* the thesis iff **all thirteen** hold (bars 1–6 always; bar 7 from SDK v3.11; bar 8 from SDK v3.12; bar 9 from SDK v3.13; bar 10 from SDK v3.14; bar 11 from SDK v3.15; bar 12 from SDK v3.16; bar 13 from SDK v3.17):

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

13. **(SDK v3.17+) End-to-end composition split.** The milestone-
   anchored harder regime (R-63-COMPOSED-TIGHT) is **provably
   insufficient for any method that closes only one of the two
   structural axes** (W14 producer-side OR W15 decoder-side; both
   limits W14-Λ-prompt and W15-Λ-budget fire on the regime). On
   R-63-naive-tight (mag-filter naive prompt + ``T_decoder = 24``)
   every capsule strategy ties FIFO at 0.000 (W16-Λ-compose); on
   R-63-COMPOSED-TIGHT (mag-filter structured prompt +
   ``T_decoder = 24``) the W14-only-budgeted baseline (FIFO-packed-
   W13) ties FIFO at 0.000 AND the W15-only-without-W14 baseline
   (AttentionAwareBundleDecoder over the naive-prompt stream) ties
   FIFO at 0.000. **The new method must wire BOTH the W14
   structured producer protocol AND the W15 attention-aware
   decoder simultaneously.** A pure W14-only method (no decoder
   budget enforcement) does NOT clear bar 13; a pure W15-only
   method (FIFO-packed decoder over a structured-prompt stream is
   permitted as the W14-only-budgeted baseline; salience pack over
   a naive-prompt stream does NOT clear bar 13). **Pre-committed
   bench property:** under structured prompt + tight ``T_decoder``,
   the composed pipeline's pack-stats block must show
   ``position_of_first_causal_claim_avg = 0.0`` AND
   ``n_with_causal_claim_kept = N`` for n_eval = N cells, AND the
   bench property must hold in every cell (mechanically verified
   by ``Phase63ComposedTightTests::test_composed_keeps_round2_specific_claim``
   and ``Phase63ComposedTightTests::test_composed_property_holds_8_of_8``).
   **Pre-committed falsifier (W16-Λ-degenerate):** under
   ``T_decoder = 2`` (below the round-2 specific claim's token
   cost), both packers collapse to 0.000; the W16-1 win is
   conditional on a budget that admits *some* but not *all* of
   the union. **Pre-committed real-LLM anchor (W16-Λ-real-replay):**
   replaying the recorded ``phase61_real_ollama_structured_qwen2_5_14b_n8.json``
   bytes through the same pipeline at ``T_decoder = 14, K_auditor = 8``
   produces ``capsule_attention_aware - capsule_layered_fifo_packed
   = +0.500`` strict gain on real-LLM bytes; the budget band where
   this holds is ``T_decoder ∈ [13, 16]``. **Honest scope:** the
   replay path is *measurement* over recorded bytes, not a fresh
   live LLM probe; W16-C-LIVE-OLLAMA is conjectural pending a live
   ``run_phase63 --extractor=ollama --prompt-mode=structured``
   probe with Mac-1 reachable.

12. **(SDK v3.16+) Joint-correctness-and-context-efficiency split.**
   The milestone-anchored harder regime (R-62-tightbudget) is
   **provably insufficient for any FIFO-packed cross-round decoder
   under decoder-side budget pressure** (W15-Λ-budget at the
   decoder-side axis: every FIFO-packed cross-round decoder in the
   SDK ties FIFO at 0.000 on R-62-tightbudget default at
   ``T_decoder = 24``; the round-2 specific-tier disambiguator falls
   past the budget by construction in arrival-order packing). The
   new method must include an explicit *capsule-aware context
   packing layer*: a salience function over admitted handoffs that
   uses (causal tier, CCK match against the elected root_cause,
   cross-role corroboration, magnitude, round index) to reorder
   AND truncate the cross-round bundle under ``T_decoder``, with
   *hypothesis preservation* so every distinct service tag's
   highest-salience (tag, role) representatives survive the pack.
   A pure FIFO-packed decoder OR a salience-greedy packer without
   hypothesis preservation does NOT clear bar 12 on SDK v3.16.
   **Pre-committed bench property:** the salience packer's pack
   must put a round-2 specific-tier claim at rank 0 of the kept
   bundle in *every* scenario where the bench property holds
   (mechanically verified by ``Phase62DefaultTests
   ::test_tightbudget_w15_keeps_round2_specific_claim``).
   **Pre-committed token-efficiency floor:** the salience pack's
   ``tokens_kept_over_input`` ratio is bounded by 1.0 (the budget is
   strict, not lossy on the kept set). **Pre-committed falsifier:**
   under no decoder-side budget pressure (R-62-saturation,
   ``T_decoder ≥ admitted union token sum``), the W15 salience pack
   is a no-op on the answer field — both salience-pack and FIFO-pack
   tie at the W13 answer (W15-Λ-budget). The W15-1 win is *not*
   observable on regimes without budget pressure — by construction.
   **Attention-shaping caveat (honest proxy).** "Attention-aware"
   uses an *honest proxy* metric — ``position_of_first_causal_claim``
   in the salience-ordered pack. We do NOT claim transformer
   attention weights are manipulated; we DO claim that under a
   real-LLM downstream consumer, putting the highest-salience
   evidence first benefits from prompt-position attention shaping
   (a well-known property of transformer attention under typical
   positional encoding regimes). This is W15's *proxy* attention
   metric, not a transformer-attention manipulation. The bar 12
   clearance condition uses this proxy directly: rank 0 of the
   kept bundle must be a specific-tier claim in every cell where
   the bench property holds.

11. **(SDK v3.15+) Producer-side ambiguity-preservation split.** The
   milestone-anchored harder regime (R-61-naive-prompt) is
   **provably insufficient for any decoder, normaliser, or admission
   policy** (W14-Λ-prompt: when the producer collapses the bench
   property upstream — by magnitude-filtering low-magnitude decoy
   events and/or by compressing round-1 toward a single best
   diagnosis — the auditor's stream loses the cross-role decoy
   corroboration AND the round-1 generic-noise property; every
   capsule strategy in the SDK ties FIFO at 0.000 on R-61-naive-
   prompt default). The new method must include an explicit
   *producer-protocol layer*: a prompt schema that (a) requires the
   producer to emit one claim per listed event with no compressed
   summarisation AND (b) splits round-1 *observation* from round-2
   *diagnosis* AND (c) optionally permits multiple-hypothesis
   listings instead of premature collapse. A pure downstream method
   (decoder / normaliser / admission policy) does NOT clear bar 11
   on SDK v3.15. **Pre-committed bench property:** under the
   structured protocol, the synthetic magnitude-filter extractor
   restores ``round1_decoy_corroborated`` ≥ 75 % across the
   pre-committed bank seeds (mechanically verified by
   ``Phase61StructuredPromptBenchTests``). **Pre-committed
   falsifier:** the same magnitude-filter extractor under the
   *naive* prompt yields ``round1_decoy_corroborated`` = 0 / N AND
   every decoder ties FIFO at 0.000 (W14-Λ-prompt; the synthetic
   counterpart of W13-Λ-real). The naive-prompt regime is the
   *structural counterexample* showing that downstream methods
   alone cannot rescue producer-side erasure.

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

For SDK v3.17 specifically, the canonical phrasing instantiates as:

> *"On R-63-COMPOSED-TIGHT (Phase-63 composed regime: multi-
> hypothesis comparable-magnitude events × magnitude-filter
> structured-prompt producer × ``T_decoder = 24``), every method
> that closes only one of the two structural axes ties FIFO at
> ``accuracy_full = 0.000`` — W14-only-budgeted (FIFO-packed-W13
> over the structured-prompt stream) at 0.000 and
> W15-only-without-W14 (AttentionAwareBundleDecoder over the
> naive-prompt stream from R-63-naive-tight) at 0.000. On
> R-63-naive-tight every capsule strategy ties FIFO at 0.000
> (W16-Λ-compose: W14-Λ-prompt and W15-Λ-budget compose
> multiplicatively). Pairing the W14 :class:`StructuredProducerProtocol`
> + W15 :class:`AttentionAwareBundleDecoder` simultaneously
> achieves ``accuracy_full = 1.000`` on R-63-COMPOSED-TIGHT
> (``K_auditor = 12, T_auditor = 256, T_decoder = 24, n_eval = 8,
> bank_seed = 11``), strictly improving over both substrate FIFO
> (+1.000) and the W14-only-budgeted baseline (+1.000), stable
> across 5/5 alternate ``bank_seed`` values (11, 17, 23, 29, 31)
> with min gap ≥ 1.000 well above the strong-bar 0.50 threshold.
> The win does not transfer to the R-63-degen-budget falsifier
> (``T_decoder = 2``, below the round-2 disambiguator's token
> cost) where both packers tie at 0.000 (W16-Λ-degenerate).
> Backward-compat preserved: R-54 / R-55 / R-56 / R-57 / R-58 /
> R-59 / R-60 / R-61 / R-62 default banks all still hit prior
> anchors; 442/442 prior tests pass byte-for-byte. The
> W16-Λ-real-replay anchor on the recorded
> ``phase61_real_ollama_structured_qwen2_5_14b_n8.json`` bytes
> at ``T_decoder = 14, K_auditor = 8`` shows
> ``capsule_attention_aware = 0.500`` and
> ``capsule_layered_fifo_packed = 0.000`` — **+0.500 strict
> gain** over the FIFO-packed-W14-only baseline on a real-LLM
> stream. The W16-1 win is conditional on (a) the structured
> producer protocol AND (b) ``T_decoder`` strictly between the
> round-2 disambiguator's token cost and the admitted union's
> token sum AND (c) the asymmetric-corroboration multi-hypothesis
> bench property; removing any one collapses to W14-Λ-prompt /
> W15-Λ-budget / W16-Λ-degenerate. The honest cap on the SDK
> v3.17 advance is the synthetic + recorded-bytes scope: a fresh
> live LLM probe (W16-C-LIVE-OLLAMA) is conjectural pending Mac-1
> reachable."*

For SDK v3.16 specifically, the canonical phrasing instantiates as:

> *"On R-62-tightbudget (Phase-62 multi-hypothesis comparable-magnitude
> events under a strict decoder-side token budget ``T_decoder = 24``),
> every FIFO-packed cross-round decoder in the SDK
> (``capsule_layered_fifo_packed`` and the implicit FIFO truncation
> any naive cross-round decoder would apply at this budget) ties
> substrate FIFO at ``accuracy_full = 0.000`` — this is the
> structural decoder-side budget limit (W15-Λ-budget: the round-2
> specific-tier disambiguator falls past the budget under
> arrival-order packing; no downstream layer can elect the correct
> specific root_cause without it). The new
> :class:`AttentionAwareBundleDecoder` (W15) — a salience-aware
> capsule context packer with hypothesis preservation — achieves
> ``accuracy_full = 1.000`` on R-62-tightbudget default
> (``K_auditor = 12``, ``T_auditor = 256``, ``T_decoder = 24``,
> ``n_eval = 8``, ``bank_seed = 11``), strictly improving over both
> substrate FIFO (+1.000) and the SDK v3.14 W13 layered decoder under
> FIFO packing (+1.000), stable across 5/5 alternate ``bank_seed``
> values (11, 17, 23, 29, 31) with min gap ≥ 1.000 well above the
> strong-bar 0.50 threshold. The W15 packer puts the round-2
> specific-tier claim at rank 0 of the kept bundle in 8/8 scenarios
> (the proxy attention signal); the FIFO packer keeps the round-2
> claim in 0/8 scenarios. The win does not transfer to the
> R-62-saturation falsifier (``T_decoder ≥ admitted union token
> sum``) where both packers tie at W13's 1.000 (W15-Λ-budget).
> Backward-compat preserved: R-54 / R-55 / R-56 / R-57 / R-58 /
> R-59 / R-60 / R-61 default banks all still hit prior anchors;
> 393/393 prior tests pass byte-for-byte; the W15 decoder with
> ``T_decoder = None`` ties the W13 layered decoder byte-for-byte on
> the answer field (W15-3)."*

For SDK v3.15 specifically, the canonical phrasing instantiates as:

> *"On R-61-naive-prompt (synthetic magnitude-filter extractor
> simulating real-LLM producer-side compression on the W13-Λ-real
> Phase-58 events), no capsule strategy in the SDK (FIFO, priority,
> coverage, W7-2 cohort, W8 corroboration, W9 multi-service, W10
> bundle decoder, W11 multi-round, W12 robust-multi-round, **and SDK
> v3.14 W13 layered-multi-round**) improves accuracy_full above
> 0.000 — this is the structural producer-side limit (W14-Λ-prompt:
> the bench property is erased upstream, no downstream layer can
> rescue it). The new ``StructuredProducerProtocol`` (SDK v3.15) — a
> prompt schema that splits observation (round 1) from diagnosis
> (round 2) and requires one claim per listed event — restores the
> bench property on the same magnitude-filter extractor; pairing it
> with W13 layered normalisation achieves accuracy_full = 1.000 on
> R-61-structured-prompt default, strictly improving over the naive-
> prompt regime (+1.000) and over substrate FIFO (+1.000), stable
> across 5/5 alternate (bank_seed, llm_seed) values, with no
> regression on R-53..R-60 / R-61-default and audit T-1..T-7
> preserved on every cell. R-61-ollama is graded on the same 4-tier
> ladder as R-60-ollama (§ 1.5)."*

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

### 1.5 R-61-ollama 4-tier grading (SDK v3.15)

R-61-ollama is the **second real-LLM probe in the programme** (after
R-60-ollama) and the **first real-LLM probe with a producer-side
protocol intervention**. Like R-60-ollama, it is graded *separately*
from R-61-structured-prompt because real-LLM behaviour is empirically
observed (not designed). The grading has four pre-committed tiers:

* **R-61-OLLAMA-A (strong real transfer).** On the redesigned
  comparable-magnitude events under the structured producer protocol,
  the bench property holds in ≥ 75 % of scenarios under the real LLM
  AND the round-1 cross-role decoy corroboration assumption holds
  in ≥ 75 % of scenarios AND the W13 layered method strictly improves
  over substrate FIFO by ≥ 0.50 pooled accuracy_full on the real-
  LLM stream (n_eval ≥ 8). Anchor: ``run_phase61(llm_mode='ollama',
  protocol_mode='structured', n_eval=8, model='qwen2.5:14b-32k',
  endpoint=...)``.
* **R-61-OLLAMA-B (partial real transfer).** Bench property holds in
  ≥ 50 % AND either (i) accuracy_full advance over FIFO ≥ 0.20 OR
  (ii) decoy corroboration is restored AND drift inside the W13
  closure is non-zero (the protocol intervention worked but the W13
  closure-widening was not the load-bearing layer; W12 already
  saturates).
* **R-61-OLLAMA-C (null real transfer; honest negative).** Bench
  property holds in < 50 % of scenarios. The structured producer
  protocol has not closed the W13-Λ-real gap; the real-LLM
  bottleneck is *not just* prompt-side discipline. The honest reading
  is that the dominant blocker is deeper than the protocol —
  examples include: the LLM compresses *across* rounds (W14-Λ-cross),
  the LLM emits drift outside the heuristic closure (intersects
  W13-4 cosmic), or the model class itself lacks the
  observation/diagnosis distinction in its training distribution.
* **R-61-OLLAMA-D (failure).** Audit breaks on any cell, OR the LLM
  endpoint fails on > 50 % of producer calls.

R-61-OLLAMA-A clears bar 11 *and* clears the historical bar 9 (W12-
C2) end-to-end and is the strongest possible empirical anchor for
the producer-protocol axis. **The honest current reading** for SDK
v3.15 is reported in
``docs/RESULTS_WEVRA_PRODUCER_AMBIGUITY.md`` § 4 — see that file for
the per-tier outcome.

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

## 2.12. Required ingredients of R-63 (SDK v3.17)

The "harder fair regime" introduced by SDK v3.17 — the *current*
strict-gain anchor and the regime where the W14 + W15 composition
is provably load-bearing — must combine:

* **All R-62 structural ingredients.** Multi-hypothesis comparable-
  magnitude events with asymmetric corroboration (decoys ≥ 2
  distinct producer roles, golds = 1 distinct role); 2 gold + 2
  decoy services per scenario; round-1 generic-noise only; round-2
  one specific-tier disambiguator with no ``service=`` token.
  Mechanically verified by ``Phase62BankShapeTests`` on the
  inherited Phase-62 bank.
* **Producer-side compression mode** (Phase-61
  :class:`MagnitudeFilteringExtractor` with the ``prompt_mode``
  toggle): the naive prompt collapses round-1 toward top-N events
  per (role, kind), erasing the bench property's
  ``all_decoys_round1_corroborated`` predicate; the structured
  prompt's per-event mandate restores it. Mechanically verified
  by the ``r63_w14_only`` cell (bench property holds in 0/8 under
  naive prompt) and the ``r63_w14_success`` cell (8/8 under
  structured prompt) of
  ``vision_mvp.experiments.phase63_composed_real_llm.run_cross_regime_summary``.
* **Decoder-side token budget** ``T_decoder`` strictly between the
  round-2 specific-tier claim's token cost and the admitted-union
  token sum. The default tight value is ``T_decoder = 24`` on
  Phase-62 events (mirroring R-62-tightbudget); the
  W16-Λ-degenerate falsifier sub-bank sets ``T_decoder = 2`` (below
  the round-2 disambiguator's cost).
* **Composed-pipeline pack-stats invariant.** Under structured
  prompt + ``T_decoder = 24``, the
  :class:`AttentionAwareBundleDecoder` pack-stats block must show
  ``position_of_first_causal_claim_avg = 0.0``,
  ``n_with_causal_claim_kept = n_eval``, and
  ``hypothesis_count_kept = hypothesis_count_input``. Mechanically
  verified by ``Phase63ComposedTightTests::test_composed_keeps_round2_specific_claim``.
* **Audit-preserving.** T-1..T-7 holds on every cell of every
  capsule strategy on every (extractor, prompt_mode, T_decoder,
  bank_seed) cell. Mechanically verified by
  ``Phase63ComposedTightTests::test_audit_OK_on_every_capsule_strategy``.
* **Determinism.** The Phase-62 bank generator is RNG-deterministic
  given ``bank_seed``; the magnitude-filter extractor is
  deterministic given ``(seed, prompt_mode)``; the W15 salience
  pack is byte-deterministic given ``(input handoffs, root_cause,
  T_decoder, round_index_hint)``. The
  :class:`OllamaReplayExtractor` is byte-deterministic given the
  recorded JSON.
* **No regression on R-54..R-62.** Every prior anchor remains
  green; the composed pipeline ties the W13 layered decoder
  byte-for-byte under ``T_decoder = None`` and identity producer;
  the composed pipeline ties the W14 anchor byte-for-byte under
  ``T_decoder = None`` and structured-prompt producer; 442/442
  prior tests pass.
* **Honest real-LLM scope.** The replay path uses recorded
  Phase-61 ``qwen2.5:14b-32k`` bytes from
  ``docs/data/phase61_real_ollama_structured_qwen2_5_14b_n8.json``
  (the W14-Λ-real anchor's source). The replay extractor is
  byte-stable; the budget band where the W16 strict gain holds on
  the recorded stream is ``T_decoder ∈ [13, 16]``; a fresh live
  LLM probe (W16-C-LIVE-OLLAMA) is conjectural pending Mac-1
  reachable.

A regime missing any of these is *not* R-63 — it does not test
the SDK v3.17 hypothesis.

## 2.11. Required ingredients of R-62 (SDK v3.16)

The "harder fair regime" introduced by SDK v3.16 — the *current*
strict-gain anchor and the regime where decoder-side context
packing is provably load-bearing — must combine:

* **All R-58 structural ingredients (post-R-58 decoder shape).**
  R-62 retains the two-round delayed-causal-evidence shape: round 1
  carries generic-noise emissions on every service tag; round 2
  carries one specific-tier disambiguating claim with no
  ``service=`` token; decoy services do not appear in round 2.
* **Multi-hypothesis decoy population.** Each scenario carries
  ``|gold_services| = 2`` AND ``|decoy_services| ≥ 2``. The total
  number of distinct service hypotheses surviving naive admission is
  ≥ 4. Mechanically verified by
  ``Phase62BankShapeTests::test_every_scenario_has_two_distinct_decoys``.
* **Asymmetric corroboration shape.** Every decoy service is
  mentioned by ≥ 2 distinct producer roles in round 1 (the W11-drop
  trigger condition); every gold service is mentioned by exactly 1
  distinct producer role in round 1. This is the load-bearing
  structural separator: it lets the W11 contradiction-aware drop
  remove decoys from the answer set without removing golds.
  Mechanically verified by
  ``Phase62BankShapeTests::test_all_decoys_corroborated_all_golds_single_role``.
* **Comparable-magnitude decoy events.** Decoy round-1 spikes have
  operational magnitudes inside the gold magnitude range so the
  magnitude-filter producer (W14) does not trivially separate them.
  R-62 inherits the W14-1 closure property from R-61.
* **Decoder-side token budget ``T_decoder``.** The R-62-tightbudget
  regime sets ``T_decoder = 24`` (per the pre-committed default), a
  value strictly below the admitted union's token sum
  (``≈ 27 / 28`` per scenario at K_auditor=12, n_eval=8, bank_seed=11).
  Under this budget, FIFO packing of the union drops the round-2
  specific-tier disambiguator by construction. Mechanically verified
  by
  ``Phase62DefaultTests::test_tightbudget_fifo_pack_drops_round2_specific_claim``.
* **Saturation-falsifier sub-bank.** R-62-saturation sets
  ``T_decoder = 9999`` (effectively unbounded). The W15-1 win
  evaporates on this regime by construction (the salience packer
  keeps every handoff regardless of budget). Mechanically verified
  by
  ``Phase62DefaultTests::test_saturation_falsifier_w15_ties_fifo_pack``.
* **Audit-preserving.** T-1..T-7 holds on every cell of every
  capsule strategy on every (T_decoder, bank_seed) cell.
* **Determinism.** The Phase-62 synthetic extractor is RNG-
  deterministic given ``(bank_seed, scenario_id)``; the W15
  salience pack is byte-deterministic given the input handoffs +
  the elected root_cause + ``T_decoder``.
* **No regression on R-54..R-61.** The W15 decoder with
  ``T_decoder = None`` ties the W13 layered decoder byte-for-byte
  on the answer field (W15-3 backward-compat anchor). On R-58
  default n=8 the W15 answers match W13 in 8/8 scenarios; on
  R-54..R-57 the admitted set is small enough that the trust-
  admission fallback fires identically under both decoders. The
  W15 ``pack_stats`` block is purely additive on prior reports.

A regime missing any of these is *not* R-62 — it does not test
the SDK v3.16 hypothesis.

## 2.10. Required ingredients of R-61 (SDK v3.15)

The "harder fair regime" introduced by SDK v3.15 — the *current*
strict-gain anchor and the regime where producer-side ambiguity-
preservation is provably load-bearing — must combine:

* **All R-58 structural ingredients.** Two-round delayed-causal-
  evidence shape, generic-noise round 1, specific-tier round 2 with
  no ``service=`` token. The bench-property post-normalisation
  witnesses (``round1_only_generic_noise``,
  ``round2_only_specific``, ``decoy_only_in_round1``,
  ``round1_decoy_corroborated``) are mechanically tested.
* **Comparable-magnitude decoy events** (R-61-default): the decoy
  storm service emits round-1 events whose operational magnitudes
  (``p95_ms``, ``error_rate``, firewall-deny counts) lie in the
  *same range* as the gold services'. Mechanically verified by
  ``Phase61BankShapeTests::test_decoy_magnitudes_within_gold_range``
  and named in code as the W14-1 closure property. A magnitude-
  filter producer that retains only events above a single
  threshold cannot distinguish gold from decoy.
* **Magnitude-filter producer simulation** (R-61-naive-prompt):
  A deterministic ``MagnitudeFilteringExtractor`` calibrated against
  the W13-Λ-real Mac-1 14B observation. On the Phase-58 (low-
  magnitude-decoy) events under the *naive* prompt, its outputs
  match the real-LLM observation in
  ``docs/data/phase60_real_ollama_qwen2_5_14b_n4.json`` byte-for-
  byte on ``round1_decoy_corroborated`` (= 0/4 in both). On the
  Phase-61 (comparable-magnitude) events it produces non-trivially
  different output depending on prompt mode (naive vs structured).
* **Structured producer protocol** (R-61-structured-prompt): the
  ``StructuredProducerProtocol`` schema renders prompts that (a)
  ask for *observation* in round 1 ("describe what you see — do
  not diagnose"), (b) ask for *diagnosis* in round 2 ("what is the
  underlying cause?"), AND (c) require one claim per listed event
  with no compression. Mechanically verified that the protocol's
  prompt rendering is byte-for-byte deterministic given the
  ``RoleExtractionSchema`` inputs.
* **Audit-preserving.** T-1..T-7 holds on every cell of every
  capsule strategy on every (LLM mode, prompt mode, bank_seed)
  cell.
* **Determinism.** The synthetic magnitude-filter extractor is
  RNG-deterministic given ``(bank_seed, llm_seed, scenario_id,
  round_idx)``. The real-Ollama extractor is best-effort
  deterministic via ``temperature=0`` and is opt-in only.
* **No regression on R-54..R-60.** The structured producer protocol
  is *additive*; on R-58 / R-59-clean / R-60-clean / R-60-wide /
  R-60-cosmic, plugging the protocol into the synthetic extractor
  produces the *same* downstream outcomes as the original prompt
  (the decoder still wins / still ties / still fails identically).

A regime missing any of these is *not* R-61 — it does not test
the SDK v3.15 hypothesis.

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
