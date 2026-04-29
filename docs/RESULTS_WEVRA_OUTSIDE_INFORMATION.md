# SDK v3.21 — outside-witness acquisition disambiguator (W20 family)

> Theory-forward results note for the post-W19 milestone. This is the
> first capsule-native multi-agent-coordination method that crosses
> the **W19-Λ-outside** wall (named in SDK v3.20) on a regime where
> the wall actually applies — by acquiring asymmetric evidence from a
> registered outside source rather than rescoring the bundle. The
> bundle-only walls (W19-Λ-total, W19-Λ-outside) remain real for
> closed-form bundle-only scorers; W20 escapes them *partially*,
> bounded by oracle integrity (W20-Λ-compromised). Date stamp:
> 2026-04-29.

## TL;DR

* **W20-1 (proved-conditional + proved-empirical, n=8 saturated × 5
  seeds × 2 cells, also n=12).** Pairing the W19
  ``BundleContradictionDisambiguator`` with the W20
  ``OutsideWitnessAcquisitionDisambiguator`` over a deterministic
  ``ServiceGraphOracle`` strictly improves ``accuracy_full`` over
  the strongest non-W20 capsule baseline by **+1.000** on
  ``R-67-OUTSIDE-RESOLVES-LOOSE`` (``T_decoder = None``) AND on
  ``R-67-OUTSIDE-RESOLVES-TIGHT`` (``T_decoder = 24``); stable
  across **5/5** alternate ``bank_seed`` values
  ``(11, 17, 23, 29, 31)``. The first capsule-native multi-agent-
  coordination method that crosses the
  **W19-Λ-outside** wall (named SDK v3.20) on a regime where the
  wall actually applies.
  Code anchor: ``vision_mvp.experiments.phase67_outside_information.run_phase67``.
  Artifact: ``docs/data/phase67_cross_regime_K12_n8.json``,
  ``docs/data/phase67_seed_sweep_loose_K12_n8.json``,
  ``docs/data/phase67_seed_sweep_tight_K12_n8.json``.
* **Three named falsifiers fire by construction.**
  ``R-67-OUTSIDE-NONE`` (``W20-Λ-none`` — abstaining oracle ⇒ ties
  FIFO 0.000), ``R-67-OUTSIDE-COMPROMISED`` (``W20-Λ-compromised`` —
  adversarial oracle ⇒ W20 picks decoy and FAILS at 0.000),
  ``R-67-JOINT-DECEPTION`` (``W20-Λ-joint-deception`` — primary +
  secondary + oracle ALL favour decoy ⇒ FAILS at 0.000). The W20-1
  conditionality is **strongly conditional** on oracle integrity AND
  on the registered oracle producing a payload whose tokenisation
  finds proper-subset asymmetric mentions of the admitted tags.
* **Honest cap on live LLM transfer (W20-Λ-real, empirical n=4).**
  A ``LLMAdjudicatorOracle`` over a fresh live Mac-1 Ollama backend
  produces *measured*, not claimed, results:
    * **qwen2.5-coder:7b** trusts the deceptive primary
      (``services=cache``); ``acc_full = 0.000``, ``+0.000`` over W19.
    * **mixtral:8x7b** (47B-MoE) free-form reply
      (``"Based on the provided information, the deadlock occurred
      between the orders and payments service"``) lands gold tokens
      asymmetrically through the W20 parser; ``acc_full = 0.750``,
      ``+0.750`` over W19 — a strict but partial live advance over
      the W19-Λ-outside wall.
  The closed-form ``ServiceGraphOracle`` is the load-bearing oracle
  on the synthetic regime; the live LLM adjudicator path is
  **probabilistic** and bounded by the closed-vocabulary closure
  W20 / W19 / W18 / W13 share. Artifacts:
  ``docs/data/phase67_live_qwen2_5_coder_7b_n4.json``,
  ``docs/data/phase67_live_mixtral_8x7b_n4.json``.
* **Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
  ``incomplete`` at milestone capture — same status as SDK v3.6
  through SDK v3.20. **No two-Mac sharded inference happened in
  SDK v3.21.** The W20 ``OutsideWitnessOracle`` Protocol is
  *infrastructure-ready* for cross-host deployment (producer roles
  on Mac 1 + adjudicator on Mac 2) when Mac 2 returns; the
  ``MLXDistributedBackend`` adapter is byte-for-byte unchanged.
  Strongest model class actually exercised: single-Mac
  ``mixtral:8x7b`` (46.7B-MoE Q4) on Mac 1 Ollama.
* **Bounded-context honesty preserved byte-for-byte.** The W20
  layer adds *exactly one* outside query per cell, bounded by
  ``max_response_tokens = 24``. Mechanically verified:
    * ``Phase67TokenBudgetHonestyTests::test_n_outside_tokens_per_cell_bounded``
      — ``outside_tokens_per_cell_avg ≤ 24`` on every cell.
    * ``Phase67TokenBudgetHonestyTests::test_w20_does_not_inflate_w15_tokens_kept``
      — ``tokens_kept_sum`` byte-for-byte identical between W19 and
      W20 on R-67-OUTSIDE-RESOLVES-TIGHT (``T_decoder = 24``). The
      W20 layer reads the same W15-packed bundle the W19 layer
      reads; outside-acquired tokens are accounted for as a strict
      **additional** cost, not a hidden inflation.
    * Total context delivered to the final decider:
      ``tokens_kept (W15) + n_outside_tokens (W20)``. On
      ``R-67-OUTSIDE-RESOLVES-TIGHT`` the average is
      ``tokens_kept ≤ 24 + outside_tokens ≤ 24 = 48``.
* **Backward-compat preserved byte-for-byte.** With
  ``enabled = False`` W20 reduces to W19 byte-for-byte
  (``Phase67BackwardCompatTests::test_w20_disabled_reduces_to_w19_byte_for_byte``).
  With ``enabled = True`` AND the inner W19 returns a non-trigger
  branch (``W19_BRANCH_INVERSION``, ``W19_BRANCH_CONFOUND_RESOLVED``,
  ``W19_BRANCH_PRIMARY_TRUSTED``), W20 also reduces to W19
  byte-for-byte
  (``Phase67BackwardCompatTests::test_w20_matches_w19_on_phase66_default_banks``).
  Full suite regression: **545 / 545 wevra tests pass** before the
  W20 milestone landed; **585 / 585** pass after (40 new W20 tests
  added).
* **Audit T-1..T-7 preserved on every cell of every regime.**
  ``Phase67DefaultConfigTests::test_w20_audit_OK_on_every_cell``.

## What changed from SDK v3.20 → v3.21 (one paragraph)

The W19 family (SDK v3.20) crossed the deceptive-ambiguity wall on
the *bundle-resolvable* case — when the bundle carries an independent
asymmetric witness from a non-canonical role (R-66-DECEIVE-NAIVE,
R-66-CONFOUND-RESOLVABLE). It also explicitly named TWO structural
limit walls *no closed-form bundle-only scorer can escape*:
**W19-Λ-total** (no asymmetric witness exists anywhere in the
bundle) and **W19-Λ-outside** (witnesses exist but are themselves
symmetric across the candidate set). SDK v3.20 named the natural
escape as **W19-C-OUTSIDE**: a scorer with access to outside
information (service-graph topology, prior reliability scores,
cross-incident historical evidence). SDK v3.21 implements the
smallest version of that escape — a deterministic, training-free
``OutsideWitnessAcquisitionDisambiguator`` that, when the inner W19
abstains via the symmetric branch, issues exactly **one** targeted
hypothesis-conditioned query to a registered ``OutsideWitnessOracle``
and projects the answer through the same closed-vocabulary scorer
W18 / W19 use on in-bundle witnesses. The escape is *partial* by
design: bounded above by oracle integrity (W20-Λ-compromised) and
by the closed-vocabulary closure W20 shares with W19 / W18 / W13.

## Theorem family W20 (minted by this milestone)

We pre-commit nine W20 statements. Five are **proved-empirical**
(saturated against pre-committed seeds), one is **proved by
inspection + mechanically-checked**, three are **conjectural** /
**proved-conditional + empirical-research**. Codebase status —
all numbered code paths land in
``vision_mvp/wevra/team_coord.py`` (``OutsideWitnessOracle``
Protocol + ``OutsideQuery`` / ``OutsideVerdict`` /
``ServiceGraphOracle`` / ``CompromisedServiceGraphOracle`` /
``AbstainingOracle`` / ``LLMAdjudicatorOracle`` /
``W20OutsideResult`` / ``OutsideWitnessAcquisitionDisambiguator``)
and ``vision_mvp/experiments/phase67_outside_information.py``
(R-67 driver + cross-regime + seed-stability sweeps).

* **W20-Λ-outside (extension)** *(proved-empirical n=8 saturated +
  structural sketch)*. W19-Λ-outside extends verbatim to
  ``R-67-OUTSIDE-REQUIRED-BASELINE`` (the same R-66-OUTSIDE-REQUIRED
  bundle shape, no oracle / abstaining oracle). Every capsule
  strategy in the SDK ties FIFO at 0.000; W19 abstains via
  ``W19_BRANCH_ABSTAINED_SYMMETRIC``; W20 sees the same admitted
  set, gets ``None`` from the oracle, and falls through.
  Mechanically verified by
  ``Phase67DefaultConfigTests::test_w20_baseline_ties_w19_at_zero``.
* **W20-1** *(proved-conditional + proved-empirical n=8 saturated × 5
  seeds × 2 cells, also n=12)*. On ``R-67-OUTSIDE-RESOLVES-LOOSE``
  and ``R-67-OUTSIDE-RESOLVES-TIGHT``, pairing the W19 inner with a
  deterministic ``ServiceGraphOracle`` strictly improves
  ``accuracy_full`` over **every** non-W20 capsule baseline (incl.
  W19) by ``+1.000``. Stable across 5/5 ``bank_seed`` values
  (``11, 17, 23, 29, 31``). The first capsule-native multi-agent-
  coordination method that crosses the W19-Λ-outside wall on a
  regime where the wall actually applies. Mechanically verified by
  ``Phase67DefaultConfigTests::test_w20_strict_win_resolves_loose``,
  ``Phase67DefaultConfigTests::test_w20_strict_win_resolves_tight``,
  ``Phase67SeedStabilityTests::test_w20_one_thousand_on_every_seed``.
  *Conditions* (any failure collapses the result):
    1. The bench property — W19 inner abstains via
       ``W19_BRANCH_ABSTAINED_SYMMETRIC``.
    2. The registered oracle's reply tokenises through the same
       closed-vocabulary closure that W19 / W18 / W13 share.
    3. The oracle's reply mentions a proper non-empty asymmetric
       subset of the admitted tag set.
* **W20-2** *(proved by inspection + mechanically-checked)*.
  Determinism + closed-form correctness. The W20 scorer is
  byte-stable: ``decode_rounds`` returns byte-for-byte identical
  answers given byte-identical inputs and a deterministic oracle;
  the per-tag outside-witness count depends only on the oracle's
  reply tokens and the admitted tag set; the projection rule is
  positive-set (every admitted tag the oracle mentions at all is
  in the projected answer; abstain on full-set hit / null hit /
  empty admitted set). Mechanically verified by
  ``W20DecoderUnitTests::test_determinism``,
  ``ServiceGraphOracleTests::test_oracle_determinism``.
* **W20-3** *(proved-empirical full programme regression,
  585/585 wevra tests pass)*. On R-54..R-66 default banks, the
  W20 method ties the W19 method byte-for-byte on the answer
  field — either via ``W20_BRANCH_NO_TRIGGER`` (W19 returns a
  non-trigger branch) or via ``W20_BRANCH_OUTSIDE_ABSTAINED``
  (oracle returns ``None``). With ``enabled = False``, the W20
  method reduces to W19 byte-for-byte. Mechanically verified by
  ``Phase67BackwardCompatTests::test_w20_matches_w19_on_phase66_default_banks``,
  ``Phase67BackwardCompatTests::test_w20_disabled_reduces_to_w19_byte_for_byte``,
  and the full pre-existing W19 / W18 / W17 / W16 / W15 / W14 /
  W13 / W12 / W11 / W10 / W9 / W8 / W7 / W6 / W5 / W4 / W3 test
  suites (442 prior wevra tests).
* **W20-Λ-none** *(proved-empirical n=8 saturated)*. On
  ``R-67-OUTSIDE-NONE``, the registered ``AbstainingOracle``
  returns ``None`` for every cell; W20 falls through to W19's
  symmetric abstention; ties FIFO at 0.000. Names the structural
  limit when the registered outside source has no signal.
  Mechanically verified by
  ``Phase67FalsifierTests::test_w20_lambda_none_ties_fifo``.
* **W20-Λ-compromised** *(proved-empirical n=8 saturated)*. On
  ``R-67-OUTSIDE-COMPROMISED``, the registered
  ``CompromisedServiceGraphOracle`` returns a decoy-asymmetric
  reply for every cell; W20 trusts the oracle and projects to the
  decoy set; FAILS at 0.000. Names the structural limit when the
  outside source itself is adversarial — the W20-1 escape is
  *partial* by design, bounded above by oracle integrity.
  Mechanically verified by
  ``Phase67FalsifierTests::test_w20_lambda_compromised_fails``.
* **W20-Λ-joint-deception** *(proved-empirical n=8 saturated)*. On
  ``R-67-JOINT-DECEPTION``, primary + secondary + oracle ALL
  favour decoy AND are internally consistent; no single-oracle
  cross-check can detect the deception; W20 ties W19 at 0.000.
  Names the structural limit when *all* evidence channels are
  jointly compromised. The W20-C-MULTI-ORACLE conjecture (multi-
  oracle aggregation) is the natural research escape; remains
  bounded above by the same closed-vocabulary closure and by
  oracle aggregation strategy. Mechanically verified by
  ``Phase67FalsifierTests::test_w20_lambda_joint_deception_fails``.
* **W20-Λ-real** *(proved-conditional + empirical-research, n=4
  saturated × 2 models)*. Real-LLM transfer of the W20 escape
  via ``LLMAdjudicatorOracle`` is conditional on the LLM emitting
  a reply whose token set finds asymmetric service mentions
  through the same closure. Empirical: on
  ``R-67-OUTSIDE-RESOLVES-LOOSE`` at ``n_eval = 4``,
  ``K_auditor = 12`` against fresh live Mac-1 Ollama:
    - ``qwen2.5-coder:7b`` trusts the deceptive primary;
      ``acc_full = 0.000``; W20-Λ-real fires the *under-scaled-
      model* failure mode (the model picks ``services=cache`` on
      every fired cell). Artifact:
      ``docs/data/phase67_live_qwen2_5_coder_7b_n4.json``.
    - ``mixtral:8x7b`` (46.7B-MoE) free-form reply happens to
      mention gold tags asymmetrically; ``acc_full = 0.750``;
      W20 strict gain over W19 = ``+0.750``. Artifact:
      ``docs/data/phase67_live_mixtral_8x7b_n4.json``.
  The cross-model split (7B vs 47B-MoE) suggests *scale + general
  knowledge* matters for the live W20-Λ-real escape; the
  deterministic ``ServiceGraphOracle`` remains the load-bearing
  escape on the synthetic anchor. The closure boundary is the
  same closed-vocabulary discipline that bounds W19 / W18 / W13 /
  W12; natural extensions are W20-C-LIVE-WITH-REGISTRY (give the
  LLM access to the service registry as part of the prompt) and
  W20-C-LEARNED (a small distilled outside-source scorer over
  capsule bundles) — both conjectural.
* **W20-C-LEARNED** *(conjectural)*. A small distilled outside-
  source scorer (e.g. a learned trust-weighted aggregator over
  oracle replies + character-bigram bundle features) outperforms
  the closed-form rule on a held-out cross-bench where the LLM
  emits free-form replies outside the synthetic exact-match
  closure. Out of scope SDK v3.21.
* **W20-C-MULTI-ORACLE** *(conjectural)*. Consulting *multiple*
  outside oracles (e.g. service-graph + change-history + on-call
  notes) and aggregating via majority / weighted vote escapes
  W20-Λ-compromised when *some* oracles remain trustworthy. The
  W20-C-MULTI-ORACLE axis is bounded above by the same closed-
  vocabulary closure; aggregation is a research move, not a
  structural escape from joint-deception (W20-Λ-joint-deception
  remains the wall on N-oracle aggregation when ALL N are
  compromised). Out of scope SDK v3.21.

## Summary table — accuracy + efficiency + evidence-acquisition

(``K_auditor = 12``, ``n_eval = 8``, default seed = 11. Each cell shows
``accuracy_full`` for that strategy on that regime; **bold** marks
the strongest method in the row.)

| Regime / oracle               | FIFO  | W15 (att-aware) | W18 (rel-compat) | W19 (bundle-contra) | **W20 (outside)** |
| -----------------------------:| :---: | :-------------: | :--------------: | :-----------------: | :---------------: |
| R-67-OUTSIDE-REQUIRED-BASELINE (no oracle) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **R-67-OUTSIDE-RESOLVES-LOOSE** (ServiceGraphOracle) | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** |
| **R-67-OUTSIDE-RESOLVES-TIGHT** (T_decoder=24, ServiceGraphOracle) | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** |
| R-67-OUTSIDE-NONE (AbstainingOracle) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R-67-OUTSIDE-COMPROMISED (CompromisedOracle) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| R-67-JOINT-DECEPTION (CompromisedOracle) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| **R-67-LIVE-MIXTRAL-8x7b** (n=4) | 0.000 | 0.000 | 0.000 | 0.000 | **0.750** |
| R-67-LIVE-QWEN-CODER-7B (n=4) | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Bounded-context efficiency (n=8, T_decoder=24,
R-67-OUTSIDE-RESOLVES-TIGHT):

| Strategy                       | tokens_kept_sum | handoffs_kept_sum | n_outside_tokens (avg per cell) | total_context_delivered |
| -----------------------------: | --------------: | ----------------: | ------------------------------: | ----------------------: |
| W18 (capsule_relational_compat)| (same as W15)   | (same as W15)     | 0                               | tokens_kept             |
| W19 (capsule_bundle_contradiction)| (same as W15)| (same as W15)     | 0                               | tokens_kept             |
| **W20 (capsule_outside_witness)** | **same as W19**| **same as W19** | **≤ 4 (deterministic) / ≤ 24 (live)** | **tokens_kept + n_outside_tokens** |

The W19 ``tokens_kept_sum`` is byte-for-byte identical to the W20
``tokens_kept_sum`` on R-67-OUTSIDE-RESOLVES-TIGHT (mechanically
verified by ``Phase67TokenBudgetHonestyTests::test_w20_does_not_inflate_w15_tokens_kept``).
The outside-acquired tokens are a strict additional, bounded-context
cost — no hidden inflation of the inner W15 pack stats.

Repeated-run stability (5 seeds, ``R-67-OUTSIDE-RESOLVES-LOOSE``):

| Seed | W19   | W20   | gap (W20 − W19) |
| ---: | ----: | ----: | --------------: |
| 11   | 0.000 | 1.000 | +1.000          |
| 17   | 0.000 | 1.000 | +1.000          |
| 23   | 0.000 | 1.000 | +1.000          |
| 29   | 0.000 | 1.000 | +1.000          |
| 31   | 0.000 | 1.000 | +1.000          |
| min  |       |       | **+1.000**      |

(Same table holds for ``R-67-OUTSIDE-RESOLVES-TIGHT`` —
``T_decoder = 24``.)

W20 branch distribution per regime (n=8):

| Regime                       | outside_resolved | outside_abstained | no_trigger |
| ---------------------------: | ---------------: | ----------------: | ---------: |
| R-67-OUTSIDE-REQUIRED-BASELINE| 0                | 8                 | 0          |
| R-67-OUTSIDE-RESOLVES        | **8**            | 0                 | 0          |
| R-67-OUTSIDE-NONE            | 0                | 8                 | 0          |
| R-67-OUTSIDE-COMPROMISED     | **8**            | 0                 | 0          |
| R-67-JOINT-DECEPTION         | **8**            | 0                 | 0          |

(The R-67-OUTSIDE-COMPROMISED + R-67-JOINT-DECEPTION rows show 8
``outside_resolved`` branch fires — but the projected service set
is the *decoy*, so the answer is wrong by oracle integrity, not
by branch logic.)

## Cross-regime regression — R-54..R-66 default banks (W20-3 anchor)

Mechanically verified by:

* ``Phase67BackwardCompatTests::test_w20_matches_w19_on_phase66_default_banks``
  — for each of ``corroborated``, ``deceive_naive``,
  ``confound_resolvable`` (R-66 default banks where W19 resolves),
  W20 with ``ServiceGraphOracle`` returns byte-for-byte identical
  ``services`` tuples to W19 on every cell. The W20 trigger does
  NOT fire on these regimes (``W19`` returns
  ``W19_BRANCH_INVERSION`` / ``W19_BRANCH_CONFOUND_RESOLVED`` /
  ``W19_BRANCH_PRIMARY_TRUSTED``).
* ``Phase67BackwardCompatTests::test_w20_disabled_reduces_to_w19_byte_for_byte``
  — with ``enabled = False`` W20 is byte-for-byte identical to W19
  even on the regime where W19 abstains.

## Two-Mac infrastructure (E)

* **Mac 1**: reachable, Ollama 11434, model list at milestone capture
  (sample of largest):
  ``mixtral:8x7b`` (46.7B-MoE Q4_0), ``gemma2:9b`` (9.2B Q4_0),
  ``qwen2.5-coder:7b`` (7.6B Q4_K_M), ``deepseek-r1:7b``
  (7.6B Q4_K_M), ``llama3.1:8b`` (8.0B Q4_K_M),
  ``lexi-coder:latest`` (8.0B Q4_K_M).
  Notably absent at this capture: the prior milestones'
  ``qwen2.5:14b-32k`` and ``qwen3.5:35b`` model tags. Cross-model
  comparison was therefore done against ``mixtral:8x7b`` (47B-MoE)
  on Mac 1 only.
* **Mac 2** (192.168.12.248): ARP ``incomplete`` at milestone
  capture — same status as SDK v3.6 through SDK v3.20. **No
  two-Mac sharded inference happened in SDK v3.21.** The
  ``MLXDistributedBackend`` adapter and the
  ``OutsideWitnessOracle`` Protocol are infrastructure-ready for
  cross-host deployment when Mac 2 returns: producer roles on Mac
  1, adjudicator on Mac 2 (or vice versa) — one HTTP client per
  oracle, one OpenAI-compatible wire shape, no SDK changes.
  Honest scope: this is *infrastructure preserved*, not a new
  cross-Mac result.

The W20 method is *naturally* a producer/adjudicator separation:
the producer roles emit handoffs into the bundle; the
``OutsideWitnessOracle`` is a strictly downstream component
consulted exactly once by the auditor when W19 abstains. A
co-located deployment (single Mac) and a cross-host deployment
(producer on Mac 1, adjudicator on Mac 2) are wire-compatible —
no W20 code changes required.

## Connection to the success criterion (Bar 17)

Bar 17 (added by SDK v3.21 to ``SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``):

> **(SDK v3.21+) Outside-information acquisition under bundle-only
> insufficiency split.** The milestone-anchored harder regime
> (R-67-OUTSIDE-RESOLVES) is **provably insufficient for any closed-
> form bundle-only scorer** (W19-Λ-outside extends verbatim to
> R-67-OUTSIDE-REQUIRED-BASELINE: every capsule strategy in the SDK
> ties FIFO at 0.000; W19 abstains via
> ``W19_BRANCH_ABSTAINED_SYMMETRIC``). The new method must include
> an explicit **outside-witness acquisition layer**: a Protocol-
> typed ``OutsideWitnessOracle`` that the auditor consults *once*
> per cell when the inner W19 returns a trigger branch, with a
> bounded ``max_response_tokens`` per call AND a positive-set
> projection rule that uses the same per-tag scorer W18 / W19 use
> on in-bundle witnesses. A pure W19 method (no outside acquisition
> step) does NOT clear bar 17 on SDK v3.21+.
>
> **Pre-committed bench property:** under R-67-OUTSIDE-RESOLVES,
> the deterministic ``ServiceGraphOracle`` returns a payload whose
> tokenisation finds a proper non-empty asymmetric subset of the
> admitted tag set in *every* cell (mechanically verified by
> ``ServiceGraphOracleTests::test_oracle_emits_gold_pair_when_admitted``,
> ``Phase67BenchPropertyTests::test_every_bank_holds_outside_required_shape``).
> **Pre-committed token-budget honesty:** the W20 layer reads only
> the W15-packed bundle (no extra capsule reads) AND records
> ``n_outside_tokens`` as a strict additional cost; the
> ``tokens_kept`` accounting is unchanged byte-for-byte from W19
> (mechanically verified by
> ``Phase67TokenBudgetHonestyTests::test_w20_does_not_inflate_w15_tokens_kept``).
> **Pre-committed falsifiers:** R-67-OUTSIDE-NONE (no signal —
> W20-Λ-none ties FIFO at 0.000), R-67-OUTSIDE-COMPROMISED
> (adversarial signal — W20-Λ-compromised picks decoy and FAILS at
> 0.000), R-67-JOINT-DECEPTION (jointly compromised — W20-Λ-joint-
> deception ties W19 at 0.000). On all three, the W20 method ties
> FIFO or fails by construction; the W20-1 conditionality is
> sharp.
> **Honest scope:** R-67 is a *synthetic* regime — the producer is
> :class:`IdentityExtractor` AND the oracle is a deterministic
> ``ServiceGraphOracle``. Real-LLM transfer of the outside-witness
> convention via ``LLMAdjudicatorOracle`` is W20-Λ-real (proved-
> conditional + empirical-research): the LLM must emit a reply
> whose token set finds asymmetric service mentions through the
> same closure W19 / W18 / W13 / W12 share; otherwise the W20
> exact-match parser misses by construction. The natural extensions
> are W20-C-LIVE-WITH-REGISTRY (LLM adjudicator with explicit
> service-registry context) and W20-C-LEARNED (small distilled
> outside-source scorer); both conjectural.

## Negative results & honest scope

* **The deterministic ``ServiceGraphOracle`` does the load-bearing
  work on the synthetic anchor.** W20-1's strict +1.000 gain is
  *because* the synthetic regime has a registered, untouched
  outside source whose reply is asymmetric across the admitted set.
  Remove the oracle ⇒ no escape (W20-Λ-none). Compromise the
  oracle ⇒ wrong escape (W20-Λ-compromised). The *mechanism*
  generalises; the *escape* is bounded above by oracle integrity.
* **Live LLM transfer is partial.** A 7B coding-tuned model
  (``qwen2.5-coder:7b``) trusts the deceptive primary on every
  fired cell of ``R-67-OUTSIDE-RESOLVES-LOOSE`` and fails at
  ``acc_full = 0.000``. A 47B-MoE general model (``mixtral:8x7b``)
  free-form reply happens to mention gold tags asymmetrically and
  achieves ``acc_full = 0.750`` (3/4 cells, 1 abstained). The
  cross-model split is a sharp *empirical* read; the W20-Λ-real
  wall is real.
* **Two-Mac path remains infrastructure-only.** Mac 2 is unreachable
  at this milestone capture — same status as the prior 14
  milestones. No SDK code change is required for cross-host
  deployment when Mac 2 returns; the W20 oracle path is
  duck-typed.
* **The W20 escape is partial by design.** The bundle-only walls
  W19-Λ-total and W19-Λ-outside remain real for *any* closed-form
  bundle-only scorer. W20 widens the scope from "bundle-only" to
  "bundle + outside source"; it does not escape *every* possible
  adversarial regime — joint-deception (R-67-JOINT-DECEPTION) is
  the named structural limit.

## Master plan / status synchronisation

* ``docs/RESEARCH_STATUS.md`` — milestone summary updated.
* ``docs/THEOREM_REGISTRY.md`` — W20 family entries minted.
* ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` — Bar 17 added,
  bar list updated to seventeen bars.
* ``docs/context_zero_master_plan.md`` — § next-frontier updated to
  name **W20-C-MULTI-ORACLE** + **W20-C-LIVE-WITH-REGISTRY** as the
  next research frontiers.
* ``docs/START_HERE.md`` — current-milestone line updated.
* ``README.md`` — milestone summary updated.
* ``CHANGELOG.md`` — SDK v3.21 entry.
* ``vision_mvp/wevra/__init__.py`` — ``SDK_VERSION = "wevra.sdk.v3.21"``.

The Wevra single-run product runtime contract is byte-for-byte
unchanged from SDK v3.20 — the W20 surface is purely additive in
the multi-agent coordination research slice (one new Protocol +
four oracle adapters + one disambiguator class).

## Tests added (40 new)

* ``ServiceGraphOracleTests`` × 6.
* ``AbstainingOracleTests`` × 1.
* ``CompromisedOracleTests`` × 2.
* ``W20DecoderUnitTests`` × 9 (trigger gating, branch determinism,
  no-oracle / disabled paths, default trigger set, branch-typing).
* ``Phase67BenchPropertyTests`` × 1 (every bank holds the
  R-66-OUTSIDE-REQUIRED shape).
* ``Phase67DefaultConfigTests`` × 5 (loose/tight strict win,
  baseline ties zero, branch counts, audit OK).
* ``Phase67SeedStabilityTests`` × 3 (loose/tight min-gap at strong
  bar, n=8 × 5 seeds at 1.000).
* ``Phase67FalsifierTests`` × 3 (none / compromised /
  joint-deception).
* ``Phase67BackwardCompatTests`` × 2 (R-66 default banks,
  ``enabled = False``).
* ``Phase67TokenBudgetHonestyTests`` × 3 (per-cell budget, no W15
  inflation, strict T_decoder).
* ``Phase67CrossRegimeSyntheticTests`` × 4 (cross-regime sweep
  invariants).

## How to reproduce

```bash
# Synthetic anchor (deterministic, ~6s):
python3 -m vision_mvp.experiments.phase67_outside_information \
    --bank outside_resolves --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# Tight composition:
python3 -m vision_mvp.experiments.phase67_outside_information \
    --bank outside_resolves --decoder-budget 24 \
    --K-auditor 12 --n-eval 8 --out -

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase67_outside_information \
    --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

# 5-seed stability sweep:
python3 -m vision_mvp.experiments.phase67_outside_information \
    --bank outside_resolves --seed-sweep \
    --K-auditor 12 --n-eval 8 --out -

# Live LLM adjudicator (Mac-1, 47B-MoE):
python3 -m vision_mvp.experiments.phase67_outside_information \
    --bank outside_resolves --live-adjudicator \
    --adjudicator-model mixtral:8x7b \
    --K-auditor 12 --n-eval 4 --out -

# Run the full W20 test suite:
python3 -m unittest vision_mvp.tests.test_wevra_outside_information

# Full SDK regression (585 tests):
python3 -m unittest discover vision_mvp/tests -p "test_wevra_*.py"
```
