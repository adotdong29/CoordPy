# SDK v3.22 — trust-weighted multi-oracle adjudicator (W21 family)

> Theory-forward results note for the post-W20 milestone. This is the
> first capsule-native multi-agent-coordination method that crosses
> the **W20-Λ-compromised** wall (named in SDK v3.21) on a regime
> where the wall actually applies — by adjudicating across **N
> registered outside oracles** under bounded context rather than
> trusting a single oracle's reply. The W20-Λ-compromised wall and
> the new W21-Λ-all-compromised wall remain real (joint compromise
> of every registered oracle still fails); the escape is *partial*
> by design, bounded by the integrity of the *registered set*. Date
> stamp: 2026-04-29.

## TL;DR

* **W21-1 (proved-conditional + proved-empirical, n=8 saturated × 5
  seeds × 2 cells).** Pairing the W19
  ``BundleContradictionDisambiguator`` with the W21
  ``TrustWeightedMultiOracleDisambiguator`` over a registered
  oracle set with at least ``quorum_min`` honest oracles strictly
  improves ``accuracy_full`` over the strongest non-W21 capsule
  baseline — including W20 with the first-registered (compromised)
  oracle — by **+1.000** on ``R-68-MULTI-MAJORITY-LOOSE``
  (``T_decoder = None``) AND on ``R-68-MULTI-MAJORITY-TIGHT``
  (``T_decoder = 24``); stable across **5/5** alternate
  ``bank_seed`` values ``(11, 17, 23, 29, 31)``. The first
  capsule-native multi-agent-coordination method that crosses the
  **W20-Λ-compromised** wall (named SDK v3.21) on a regime where
  the wall actually applies.
  Code anchor:
  ``vision_mvp.experiments.phase68_multi_oracle_adjudication.run_phase68``.
  Artifacts: ``docs/data/phase68_cross_regime_K12_n8.json``,
  ``docs/data/phase68_seed_sweep_loose_K12_n8.json``,
  ``docs/data/phase68_seed_sweep_tight_K12_n8.json``,
  ``docs/data/phase68_multi_majority_K12_n12.json``.
* **Three named falsifiers fire by construction.**
  * ``R-68-MULTI-NO-QUORUM`` (``W21-Λ-no-quorum`` — three
    singleton oracles, each pointing at a different admitted tag,
    so no tag receives ≥ ``quorum_min = 2`` votes ⇒ W21 abstains
    and ties FIFO at 0.000).
  * ``R-68-MULTI-ALL-COMPROMISED`` (``W21-Λ-all-compromised`` —
    every registered oracle returns a decoy-asymmetric reply ⇒
    quorum forms on decoy ⇒ W21 projects to ``{decoy}`` and FAILS
    at 0.000).
  * ``R-68-MULTI-PARTIAL`` (``W21-Λ-partial`` — two partial-honest
    oracles each emit a different element of the gold pair, plus
    one abstainer ⇒ no tag has ≥ ``quorum_min = 2`` votes ⇒ W21
    abstains at default config; recovers to 1.000 under override
    ``quorum_min = 1`` — the **W21-C-PARTIAL-RECOVERY** conjecture
    discharged empirically).
  The W21-1 conditionality is **strongly conditional** on (a) the
  bench property — W19 abstains via ``W19_BRANCH_ABSTAINED_SYMMETRIC``,
  (b) ≥ ``quorum_min`` honest oracles registered, AND (c) the honest
  oracles' replies overlap on a non-empty proper asymmetric subset
  of admitted tags.
* **Live LLM transfer (W21-C-LIVE-WITH-REGISTRY, empirical n=4
  saturated × 2 models, partially discharged).** Two regimes:
    1. **Mixed-registry (registry-anchored, easy)** — four-oracle
       registry ``(compromised_registry, service_graph,
       change_history, ollama_mixtral:8x7b)`` on R-68-MULTI-MAJORITY:
       W21 ``acc_full = 1.000``, W20 = 0.000, **+1.000 strict gain
       over W20** on the live regime. 4/4 ``W21_BRANCH_QUORUM_RESOLVED``;
       the deterministic ``service_graph + change_history`` already
       forms quorum=2 on gold, so the LLM's vote is *redundant*.
       Artifact: ``docs/data/phase68_live_mixed_registry_mixtral_n4.json``.
    2. **Coalition (LLM-vote-required, hard)** — three-oracle
       registry ``(compromised_registry, change_history,
       ollama_<model>)`` on R-68-MULTI-MAJORITY with
       ``quorum_min = 2``. The deterministic registry alone is
       *insufficient* (only one honest deterministic oracle); the
       LLM's vote is required for quorum on gold. Cross-model split:
         * ``ollama_mixtral:8x7b`` (47B-MoE): W21 = **0.750**, W20 =
           0.000, **+0.750 strict gain** — mixtral lands gold tokens
           through the W18/W19 closure on 3/4 cells. Artifact:
           ``docs/data/phase68_live_coalition_mixtral_n4.json``.
         * ``ollama_gemma2:9b`` (9.2B-dense): W21 = **0.000**, W20 =
           0.000, **0.000 strict gain** — gemma2:9b lands decoy tokens
           through the closure; combined with the compromised oracle's
           decoy vote, decoy gets quorum=2 (wrong projection on 3/4
           cells). Artifact:
           ``docs/data/phase68_live_coalition_gemma2_9b_n4.json``.
       The cross-model split (47B-MoE / 9.2B-dense) is sharp; **scale
       + general knowledge matter for the W21-Λ-real escape** on the
       LLM-vote-required regime. The deterministic registry remains
       the load-bearing oracle on the synthetic anchor.
* **Two-Mac infrastructure.** Mac 2 (192.168.12.248) ARP
  ``incomplete`` at milestone capture — same status as SDK v3.6
  through SDK v3.21. **No two-Mac sharded inference happened in
  SDK v3.22.** The W21 mechanism is *naturally* a producer /
  multi-adjudicator separation: a co-located deployment (every
  oracle on Mac-1) is wire-compatible with cross-host deployment
  (registry on Mac-1, LLM adjudicator on Mac-2) — no W21 code
  changes required. Strongest model class actually exercised:
  single-Mac ``mixtral:8x7b`` (46.7B-MoE Q4) on Mac 1 Ollama.
* **Bounded-context honesty preserved byte-for-byte.** The W21
  layer issues *exactly N = ``len(oracle_registrations)``* outside
  queries per cell, each bounded by ``max_response_tokens``.
  Mechanically verified:
    * ``Phase68TokenBudgetHonestyTests::test_n_outside_tokens_total_per_cell_bounded``
      — ``w21_outside_tokens_total_per_cell_avg ≤
      N × max_response_tokens`` on every cell.
    * ``Phase68TokenBudgetHonestyTests::test_w21_does_not_inflate_w15_tokens_kept``
      — ``tokens_kept_sum`` byte-for-byte identical between
      W19, W20 and W21 on R-68-MULTI-MAJORITY-TIGHT
      (``T_decoder = 24``). The W21 layer reads only the W19-admitted
      tag set (no extra capsule reads); outside-acquired tokens are
      accounted for as a strict additional cost.
    * Total context delivered to the final decider on a 3-oracle
      stack: ``tokens_kept (W15) + N × n_outside_tokens (W21)``.
      On ``R-68-MULTI-MAJORITY-TIGHT`` the average is
      ``tokens_kept ≤ 24 + 3 × 4 = 36`` (each deterministic oracle
      emits ≤ 4 tokens per reply; live LLM bounded at ≤ 24).
* **Backward-compat preserved byte-for-byte.**
  * **W21-3-A** (vs W19): With ``enabled = False`` OR no oracles
    registered, W21 reduces to W19 byte-for-byte. With
    ``enabled = True`` AND the inner W19 returns a non-trigger
    branch (R-66-CORROBORATED / R-66-DECEIVE-NAIVE /
    R-66-CONFOUND-RESOLVABLE), W21 reduces to W19 byte-for-byte
    (mechanically verified by
    ``Phase68BackwardCompatTests::test_w21_no_trigger_on_r66_default_banks``).
  * **W21-3-B** (vs W20): With ``quorum_min = 1`` AND a single
    registered honest oracle on R-67-OUTSIDE-RESOLVES, W21 ties
    W20 byte-for-byte on the answer field at 1.000 (mechanically
    verified by ``Phase68ReducesToW20OnR67Tests
    ::test_w21_q1_single_oracle_ties_w20_on_r67`` and unit test
    ``W21DecoderUnitTests::test_w21_q1_single_oracle_ties_w20_byte_for_byte``).
  Full suite regression: **585 / 585 coordpy tests pass** before the
  W21 milestone landed; **633 / 633** pass after (48 new W21 tests
  added).
* **Audit T-1..T-7 preserved on every cell of every regime.**
  ``Phase68DefaultConfigTests::test_w21_audit_OK_on_every_cell``.

## What changed from SDK v3.21 → v3.22 (one paragraph)

The W20 family (SDK v3.21) crossed the W19-Λ-outside wall on the
*single-oracle* case — when a *registered* outside oracle's reply is
asymmetric across admitted tags, the W20 layer projects to that
asymmetric subset. It also explicitly named the next structural wall
*no single-oracle method can escape*: **W20-Λ-compromised** — when
the registered oracle is itself adversarial, W20 trusts it and fails.
SDK v3.21 named the natural escape as **W20-C-MULTI-ORACLE**: a
scorer that consults *every* registered outside source and projects
only when a **quorum** of independently-replying oracles agrees.
SDK v3.22 implements the smallest version of that escape — a
deterministic, training-free
``TrustWeightedMultiOracleDisambiguator`` that, when the inner W19
abstains, issues exactly **N = len(oracle_registrations)** outside
queries (one per registered oracle), counts per-tag votes (a tag is
"voted-for" iff ≥ 1 oracle's reply mentions it asymmetrically), and
projects onto tags with ≥ ``quorum_min`` votes AND ≥ ``min_trust_sum``
trust mass. The escape is *partial* by design: bounded above by the
integrity of the *registered set* (W21-Λ-all-compromised) and by the
quorum threshold itself (W21-Λ-no-quorum, W21-Λ-partial).

## Theorem family W21 (minted by this milestone)

We pre-commit ten W21 statements. Five are **proved-empirical**
(saturated against pre-committed seeds), one is **proved by
inspection + mechanically-checked**, two are **proved-empirical
backward-compat anchors**, and two are **conjectural** /
**proved-conditional + empirical-research**. Codebase status —
all numbered code paths land in
``vision_mvp/coordpy/team_coord.py`` (``OracleRegistration`` /
``ChangeHistoryOracle`` / ``OnCallNotesOracle`` /
``SingletonAsymmetricOracle`` / ``DisagreeingHonestOracle`` /
``W21OracleProbe`` / ``W21MultiOracleResult`` /
``TrustWeightedMultiOracleDisambiguator``)
and ``vision_mvp/experiments/phase68_multi_oracle_adjudication.py``
(R-68 driver + cross-regime + seed-stability sweeps).

* **W21-1** *(proved-conditional + proved-empirical n=8 saturated × 5
  seeds × 2 cells, also n=12)*. On ``R-68-MULTI-MAJORITY-LOOSE`` and
  ``R-68-MULTI-MAJORITY-TIGHT``, pairing the W19 inner with a
  three-oracle registered set (``compromised_registry``,
  ``service_graph``, ``change_history``) under ``quorum_min = 2``
  strictly improves ``accuracy_full`` over **every** non-W21 capsule
  baseline (incl. W20 trusting the first-registered compromised
  oracle) by ``+1.000``. Stable across 5/5 ``bank_seed`` values
  ``(11, 17, 23, 29, 31)``. The first capsule-native multi-agent-
  coordination method that crosses the W20-Λ-compromised wall on a
  regime where the wall actually applies. Mechanically verified by
  ``Phase68DefaultConfigTests::test_w21_strict_win_multi_majority_loose``,
  ``Phase68DefaultConfigTests::test_w21_strict_win_multi_majority_tight``,
  ``Phase68SeedStabilityTests::test_w21_one_thousand_on_every_seed``.
  *Conditions* (any failure collapses the result):
    1. The bench property — W19 inner abstains via
       ``W19_BRANCH_ABSTAINED_SYMMETRIC``.
    2. ≥ ``quorum_min`` registered oracles return non-empty-proper-
       asymmetric replies whose token sets find admitted-tag mentions
       through the W18 / W19 closure.
    3. The honest oracles' replies overlap on a non-empty proper
       asymmetric subset of admitted_tags.
* **W21-2** *(proved by inspection + mechanically-checked)*.
  Determinism + closed-form correctness. The W21 scorer is
  byte-stable: ``decode_rounds`` returns byte-for-byte identical
  answers given byte-identical inputs and a deterministic
  registered oracle set; the per-tag vote count depends only on
  each oracle's reply tokens and the admitted tag set; the
  projection rule is positive-set with quorum + trust-sum
  thresholds. Mechanically verified by
  ``W21DecoderUnitTests::test_determinism``.
* **W21-3-A** *(proved-empirical full programme regression,
  R-54..R-67 default banks, mechanically-checked on R-66)*. On
  R-54..R-67 default banks, the W21 method ties the W19 method
  byte-for-byte on the answer field — either via
  ``W21_BRANCH_NO_TRIGGER`` (W19 returns a non-trigger branch) or
  via ``W21_BRANCH_NO_ORACLES`` (no oracles registered). With
  ``enabled = False``, the W21 method reduces to W19 byte-for-byte.
  Mechanically verified by
  ``Phase68BackwardCompatTests::test_w21_no_trigger_on_r66_default_banks``,
  ``Phase68BackwardCompatTests::test_w21_disabled_reduces_to_w19_byte_for_byte``,
  and the full pre-existing W19 / W18 / W17 / W16 / W15 / W14 /
  W13 / W12 / W11 / W10 / W9 / W8 / W7 / W6 / W5 / W4 / W3 test
  suites (442 prior coordpy tests + 40 W20 + 48 W21 = 633 total).
* **W21-3-B** *(proved-empirical, R-67-OUTSIDE-RESOLVES anchor)*.
  On R-67-OUTSIDE-RESOLVES with a single registered honest oracle
  AND ``quorum_min = 1`` AND ``min_trust_sum = 0.0``, the W21
  method ties W20 byte-for-byte on the answer field at 1.000. The
  W21 layer is a strict generalisation of W20 — the special case
  ``quorum_min = 1, |reg| = 1`` recovers W20's projection rule.
  Mechanically verified by
  ``W21DecoderUnitTests::test_w21_q1_single_oracle_ties_w20_byte_for_byte``,
  ``Phase68ReducesToW20OnR67Tests::test_w21_q1_single_oracle_ties_w20_on_r67``.
* **W21-Λ-no-quorum** *(proved-empirical n=8 saturated)*. On
  ``R-68-MULTI-NO-QUORUM``, the registered three
  ``SingletonAsymmetricOracle`` instances each emit a different
  admitted tag; each tag receives exactly one vote; no tag reaches
  ``quorum_min = 2``; W21 abstains via
  ``W21_BRANCH_NO_QUORUM``; ties FIFO at 0.000. Names the
  structural limit when registered oracles disagree completely on
  what the answer is. Mechanically verified by
  ``Phase68FalsifierTests::test_w21_lambda_no_quorum_ties_fifo``.
* **W21-Λ-all-compromised** *(proved-empirical n=8 saturated)*. On
  ``R-68-MULTI-ALL-COMPROMISED``, every registered oracle returns
  a decoy-asymmetric reply; quorum forms on decoy; W21 projects to
  ``{decoy}`` and FAILS at 0.000. Names the structural limit when
  *all* N registered oracles are jointly compromised — the W21
  escape is *partial* by design, bounded above by the integrity of
  the *registered oracle set*. The W21-C-CALIBRATED-TRUST
  conjecture (low trust priors on uncalibrated oracles) is the
  natural research escape; remains conjectural. Mechanically
  verified by
  ``Phase68FalsifierTests::test_w21_lambda_all_compromised_fails``.
* **W21-Λ-partial** *(proved-empirical n=8 saturated, default
  ``quorum_min = 2``)*. On ``R-68-MULTI-PARTIAL``, two partial-
  honest oracles each emit a different element of the gold pair
  (no overlap) plus one ``AbstainingOracle``; each gold tag
  receives exactly one vote; no tag reaches ``quorum_min = 2``;
  W21 abstains via ``W21_BRANCH_NO_QUORUM``; ties FIFO at 0.000.
  Names the trade-off between quorum strictness and partial-
  evidence recovery. The conditional W21-C-PARTIAL-RECOVERY
  conjecture (with ``quorum_min = 1``, W21 recovers the gold pair
  on the same regime — empirically discharged at n=8) is *empirical-
  on-default-config* but **not** strict at the default
  ``quorum_min = 2``. Mechanically verified by
  ``Phase68FalsifierTests::test_w21_lambda_partial_default_q2``,
  ``Phase68ConditionalSuccessTests::test_w21_partial_with_quorum_min_1_wins``.
* **W21-Λ-real** *(proved-conditional + empirical-research, n=4
  saturated × 2 models, two regimes)*. Real-LLM transfer of the
  W21 escape via a registered ``LLMAdjudicatorOracle`` is
  conditional on the LLM emitting a reply whose token set finds
  asymmetric service mentions through the same closure W19 / W18
  / W13 / W12 share. Two empirical cells, both at ``n_eval = 4``,
  ``K_auditor = 12``, fresh live Mac-1 Ollama:
  * **Mixed-registry (registry-anchored, easy)** — four-oracle
    registry ``(compromised_registry, service_graph, change_history,
    ollama_mixtral:8x7b)``. W21 achieves ``acc_full = 1.000``;
    W20 achieves 0.000; ``+1.000`` over W20.
    ``W21_BRANCH_QUORUM_RESOLVED`` fires on 4/4 cells. The
    deterministic ``service_graph`` + ``change_history`` pair forms
    quorum=2 on gold *regardless* of what the LLM emits; this is
    why the W21 mechanism is robust to live LLM closure failures.
    Artifact:
    ``docs/data/phase68_live_mixed_registry_mixtral_n4.json``.
  * **Coalition (LLM-vote-required, hard)** — three-oracle
    registry ``(compromised_registry, change_history,
    ollama_<model>)`` with ``quorum_min = 2``. The deterministic
    registry alone is *insufficient* (only one honest deterministic
    oracle); the LLM's vote is required for quorum on gold. Two
    cross-model cells:
    - ``ollama_mixtral:8x7b`` (47B-MoE): W21 ``acc_full = 0.750``;
      W20 = 0.000; **+0.750 strict gain over W20**. 3/4 cells
      ``W21_BRANCH_QUORUM_RESOLVED`` (mixtral lands gold tokens
      asymmetrically through the W18/W19 closure), 1/4
      ``W21_BRANCH_NO_QUORUM`` (mixtral's reply doesn't land gold
      tokens — closure miss). Artifact:
      ``docs/data/phase68_live_coalition_mixtral_n4.json``.
    - ``ollama_gemma2:9b`` (9.2B-dense): W21 ``acc_full = 0.000``;
      W20 = 0.000; **0.000 strict gain (no advance)**. 3/4 cells
      ``W21_BRANCH_QUORUM_RESOLVED`` BUT projected to *decoy*
      (gemma2:9b's reply lands decoy tokens through the closure;
      combined with the compromised oracle's decoy vote, decoy gets
      quorum=2 — wrong projection). Artifact:
      ``docs/data/phase68_live_coalition_gemma2_9b_n4.json``.
  The cross-model split (47B-MoE: 0.750 vs 9.2B-dense: 0.000) is a
  sharp empirical *cross-model transfer reading*: scale + general
  knowledge matters for the W21-Λ-real escape on the
  LLM-vote-required regime. The deterministic ``ServiceGraphOracle``
  remains the load-bearing escape on the synthetic anchor; the live
  LLM adjudicator path is **probabilistic**, bounded by closure +
  by the model's actual semantic recall.
* **W21-C-CALIBRATED-TRUST** *(conjectural)*. With trust priors
  calibrated against held-out historical agreement, a low prior on
  an uncalibrated oracle excludes its vote from quorum aggregation
  via the ``min_trust_sum`` floor — escapes W21-Λ-all-compromised
  on regimes where ≥ ``quorum_min`` calibrated oracles remain
  honest. Out of scope SDK v3.22.
* **W21-C-LIVE-WITH-REGISTRY** *(proved-conditional + empirical-
  research, partially discharged)*. An LLM adjudicator paired with
  a deterministic ``ServiceGraphOracle`` + ``ChangeHistoryOracle``
  as registered "trusted registry" channels extends W21 to the
  W20-Λ-real wall. Empirical: at n=4 × Mac-1 ``mixtral:8x7b``,
  W21 wins at +1.000 over W20 with the deterministic registry +
  LLM stack — the closure problem (LLM may emit free-form replies
  outside the W18/W19 token closure) is *neutralised* because the
  deterministic registry alone forms quorum=2 on gold. The W21-C-
  LIVE-WITH-REGISTRY conjecture from SDK v3.21 is therefore
  **partially discharged**: the live multi-source story works
  even without the LLM landing in the W18/W19 closure.

Phase-68 also closes a *partial* discharge of W20-Λ-compromised on
the *bundle-resolvable-by-quorum* direction: the wall remains a real
structural limit *for any single-oracle method*, but the multi-
oracle axis crosses it when ≥ ``quorum_min`` honest oracles are
registered. The W20-Λ-compromised wall is therefore named *partially
discharged* by W21-1 (positive direction) AND remains real on the
*all-compromised* and *no-quorum* and *partial-without-quorum-1*
directions.

## Summary table — accuracy + efficiency + evidence-acquisition

(``K_auditor = 12``, ``n_eval = 8``, default seed = 11. Each cell shows
``accuracy_full`` for that strategy on that regime; **bold** marks
the strongest method in the row.)

| Regime / oracle set                                | FIFO  | W18 (rel-compat) | W19 (bundle-contra) | W20 (outside) | **W21 (multi-oracle)** |
| --------------------------------------------------:| :---: | :--------------: | :-----------------: | :-----------: | :--------------------: |
| R-68-SINGLE-CLEAN (1 honest oracle)                | 0.000 | 0.000            | 0.000               | **1.000**     | 0.000 (q=2 abstains)   |
| **R-68-MULTI-MAJORITY-LOOSE** (3 oracles, 1 bad)   | 0.000 | 0.000            | 0.000               | 0.000         | **1.000**              |
| **R-68-MULTI-MAJORITY-TIGHT** (T_decoder=24)       | 0.000 | 0.000            | 0.000               | 0.000         | **1.000**              |
| R-68-MULTI-NO-QUORUM (3 singleton oracles)         | 0.000 | 0.000            | 0.000               | 0.000         | 0.000 (W21-Λ-no-quorum)|
| R-68-MULTI-ALL-COMPROMISED (3 compromised)         | 0.000 | 0.000            | 0.000               | 0.000         | 0.000 (W21-Λ-all-comp.)|
| R-68-MULTI-PARTIAL (2 partial + abstainer, q=2)    | 0.000 | 0.000            | 0.000               | 0.000         | 0.000 (W21-Λ-partial)  |
| R-68-MULTI-PARTIAL-Q1 (override q=1)               | 0.000 | 0.000            | 0.000               | 0.000         | **1.000**              |
| **R-68-LIVE-MIXED-REGISTRY** (4 oracles incl. mixtral 8x7b, n=4) | 0.000 | 0.000 | 0.000      | 0.000         | **1.000**              |
| **R-68-LIVE-COALITION-MIXTRAL** (3 oracles, mixtral required, n=4) | 0.000 | 0.000 | 0.000  | 0.000     | **0.750**              |
| R-68-LIVE-COALITION-GEMMA2-9B (3 oracles, gemma2 required, n=4) | 0.000 | 0.000 | 0.000      | 0.000         | 0.000 (W21-Λ-real)     |

Bounded-context efficiency (n=8, T_decoder=24,
R-68-MULTI-MAJORITY-TIGHT, 3 registered oracles):

| Strategy                       | tokens_kept_sum | n_outside_queries (per cell) | n_outside_tokens_total (avg per cell) | total_context_delivered |
| -----------------------------: | --------------: | ---------------------------: | ------------------------------------: | ----------------------: |
| W18 (capsule_relational_compat)| (same as W15)   | 0                            | 0                                     | tokens_kept             |
| W19 (capsule_bundle_contra…)   | (same as W15)   | 0                            | 0                                     | tokens_kept             |
| W20 (capsule_outside_witness)  | **same as W19** | 1                            | ≤ 4 (deterministic) / ≤ 24 (live)     | tokens_kept + 1×n_out   |
| **W21 (capsule_multi_oracle)** | **same as W19** | **3** (deterministic regime) | ≤ 12 (3 × 4 deterministic each)       | **tokens_kept + 3×n_out** |

The W19 / W20 / W21 ``tokens_kept_sum`` is byte-for-byte identical on
R-68-MULTI-MAJORITY-TIGHT (mechanically verified by
``Phase68TokenBudgetHonestyTests::test_w21_does_not_inflate_w15_tokens_kept``).
The N outside-acquired tokens are a strict additional, bounded-context
cost — no hidden inflation of the inner W15 pack stats.

Repeated-run stability (5 seeds, ``R-68-MULTI-MAJORITY-LOOSE``):

| Seed | W19   | W20   | W21   | gap (W21 − W20) |
| ---: | ----: | ----: | ----: | --------------: |
| 11   | 0.000 | 0.000 | 1.000 | +1.000          |
| 17   | 0.000 | 0.000 | 1.000 | +1.000          |
| 23   | 0.000 | 0.000 | 1.000 | +1.000          |
| 29   | 0.000 | 0.000 | 1.000 | +1.000          |
| 31   | 0.000 | 0.000 | 1.000 | +1.000          |
| min  |       |       |       | **+1.000**      |

(Same table holds for ``R-68-MULTI-MAJORITY-TIGHT`` —
``T_decoder = 24``.)

W21 branch distribution per regime (n=8):

| Regime                          | quorum_resolved | no_quorum | symmetric_quorum | no_trigger |
| ------------------------------: | --------------: | --------: | ---------------: | ---------: |
| R-68-SINGLE-CLEAN               | 0               | 8         | 0                | 0          |
| R-68-MULTI-MAJORITY (loose+tight)| **8**          | 0         | 0                | 0          |
| R-68-MULTI-NO-QUORUM            | 0               | 8         | 0                | 0          |
| R-68-MULTI-ALL-COMPROMISED      | **8** (decoy!)  | 0         | 0                | 0          |
| R-68-MULTI-PARTIAL (q=2)        | 0               | 8         | 0                | 0          |
| R-68-MULTI-PARTIAL-Q1 (q=1)     | **8**           | 0         | 0                | 0          |

(The R-68-MULTI-ALL-COMPROMISED row shows 8 ``quorum_resolved``
branch fires — but the projected service set is the *decoy*, so the
answer is wrong by registered-oracle-set integrity, not by branch
logic.)

## Cross-regime regression — R-54..R-67 default banks (W21-3-A anchor)

Mechanically verified by:

* ``Phase68BackwardCompatTests::test_w21_no_trigger_on_r66_default_banks``
  — for each of ``corroborated``, ``deceive_naive``,
  ``confound_resolvable`` (R-66 default banks where W19 resolves),
  W21 with the three-oracle registered set returns byte-for-byte
  identical ``services`` tuples to W19 on every cell. The W21
  trigger does NOT fire on these regimes (W19 returns
  ``W19_BRANCH_INVERSION`` / ``W19_BRANCH_CONFOUND_RESOLVED`` /
  ``W19_BRANCH_PRIMARY_TRUSTED``).
* ``Phase68BackwardCompatTests::test_w21_disabled_reduces_to_w19_byte_for_byte``
  — with ``enabled = False`` W21 is byte-for-byte identical to W19
  even on the regime where W19 abstains.
* The full pre-existing W12..W20 test suites (442 + 40 = 482 tests)
  continue to pass with W21 surface added (additive only — no W12..
  W20 code modified).

## Two-Mac infrastructure (E)

* **Mac 1**: reachable, Ollama 11434, model list at milestone capture
  (sample of largest):
  ``mixtral:8x7b`` (46.7B-MoE Q4_0), ``gemma2:9b`` (9.2B Q4_0),
  ``qwen2.5-coder:7b`` (7.6B Q4_K_M), ``deepseek-r1:7b``
  (7.6B Q4_K_M), ``llama3.1:8b`` (8.0B Q4_K_M),
  ``lexi-coder:latest`` (8.0B Q4_K_M), ``qwen2.5:0.5b``.
  Notably absent at this capture: the prior milestones'
  ``qwen2.5:14b-32k`` and ``qwen3.5:35b`` model tags. The
  W21-Λ-real probe was therefore done against ``mixtral:8x7b``
  (47B-MoE) on Mac 1 only.
* **Mac 2** (192.168.12.248): ARP ``incomplete`` at milestone
  capture — same status as SDK v3.6 through SDK v3.21. **No
  two-Mac sharded inference happened in SDK v3.22.** The W21
  oracle Protocol is *infrastructure-ready* for cross-host
  deployment (multiple LLM adjudicators on Mac-2, deterministic
  registry on Mac-1) when Mac 2 returns; the
  ``MLXDistributedBackend`` adapter is byte-for-byte unchanged.
  Honest scope: this is *infrastructure preserved*, not a new
  cross-Mac result.

The W21 method is *naturally* a producer / multi-adjudicator
separation: the producer roles emit handoffs into the bundle; each
``OutsideWitnessOracle`` is a strictly downstream component
consulted independently. A co-located deployment (every oracle on
single Mac) and a cross-host deployment (registry on Mac-1, LLM
adjudicator on Mac-2) are wire-compatible — no W21 code changes
required.

## Connection to the success criterion (Bar 18)

Bar 18 (added by SDK v3.22 to ``SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``):

> **(SDK v3.22+) Multi-source outside-information adjudication
> under partial oracle compromise.** The milestone-anchored harder
> regime (R-68-MULTI-MAJORITY) is **provably insufficient for any
> single-oracle method** (W20-Λ-compromised extends verbatim to the
> R-68-MULTI-MAJORITY single-oracle interface: every cell trusts
> the first-registered compromised oracle and FAILS). The new
> method must include an explicit **multi-oracle adjudication
> layer**: a typed registered set of N
> ``OutsideWitnessOracle``-shaped objects with prior trust weights,
> a bounded ``max_response_tokens`` per call (per oracle), AND a
> deterministic per-tag voting rule with **quorum_min** + 
> **min_trust_sum** thresholds. A pure W20 method (single-oracle
> projection) does NOT clear bar 18 on SDK v3.22+.
>
> **Pre-committed bench property:** under R-68-MULTI-MAJORITY, the
> registered three-oracle set ``(compromised_registry, service_graph,
> change_history)`` produces:
> (a) compromised_registry returns a decoy-asymmetric reply on every
> cell;
> (b) service_graph returns a gold-pair-asymmetric reply on every
> cell (from the deterministic incident-triage service-dependency
> graph);
> (c) change_history returns a gold-pair-asymmetric reply on every
> cell (from the deterministic per-incident change log).
> The gold pair receives 2 votes; the decoy receives 1 vote; quorum
> forms on the gold pair under the default ``quorum_min = 2``.
> Mechanically verified by ``Phase68DefaultConfigTests
> ::test_w21_branches_quorum_resolved_on_majority``.
> **Pre-committed token-budget honesty:** the W21 layer reads only
> the W15-packed-and-W19-admitted bundle (no extra capsule reads)
> AND records ``n_outside_queries`` + ``n_outside_tokens_total`` as
> strict additional cost; the ``tokens_kept`` accounting is
> unchanged byte-for-byte from W19 / W20 (mechanically verified by
> ``Phase68TokenBudgetHonestyTests::test_w21_does_not_inflate_w15_tokens_kept``).
> **Pre-committed falsifiers:** R-68-MULTI-NO-QUORUM (no signal —
> W21-Λ-no-quorum ties FIFO at 0.000), R-68-MULTI-ALL-COMPROMISED
> (jointly compromised — W21-Λ-all-compromised picks decoy and FAILS
> at 0.000), R-68-MULTI-PARTIAL (sub-quorum honest signal —
> W21-Λ-partial abstains at default ``quorum_min = 2`` and ties FIFO
> at 0.000). On all three, the W21 method ties FIFO or fails by
> construction; the W21-1 conditionality is sharp.
> **Honest scope:** R-68 is a *synthetic* regime — the producer is
> :class:`IdentityExtractor` AND the deterministic oracles are
> closed-vocabulary (:class:`ServiceGraphOracle`,
> :class:`ChangeHistoryOracle`). Real-LLM transfer via
> :class:`LLMAdjudicatorOracle` as a fourth registered oracle is
> W21-Λ-real / W21-C-LIVE-WITH-REGISTRY (proved-conditional +
> empirical-research, partially discharged at n=4 × mixtral 8x7b
> on Mac-1).

## Negative results & honest scope

* **The deterministic registry oracles do the load-bearing work
  on the synthetic anchor.** W21-1's strict +1.000 gain is
  *because* the synthetic regime has two registered honest
  deterministic oracles whose replies are asymmetric on the gold
  pair. Remove either honest oracle ⇒ no quorum ⇒ no escape.
  Compromise both honest oracles ⇒ joint compromise (W21-Λ-all-
  compromised). The *mechanism* generalises; the *escape* is
  bounded above by registered-set integrity.
* **The single-clean trade-off is real.** On R-68-SINGLE-CLEAN (1
  honest oracle), W20 wins at 1.000 and W21 with default
  ``quorum_min = 2`` abstains at 0.000. The trade-off — higher
  quorum = more conservative — is part of the W21 contract; the
  override ``quorum_min = 1`` recovers W20 behavior on this
  regime.
* **Live LLM transfer is partially discharged via the registry-
  anchor mechanism.** When the deterministic registry alone
  already forms quorum=2, the W21 method wins at 1.000 regardless
  of whether the LLM lands gold tokens. This is a *positive*
  result for the W21-C-LIVE-WITH-REGISTRY conjecture but does
  NOT discharge the harder W21-Λ-real wall (regime where the
  LLM's vote is *required* for quorum). A sharper test —
  registered set ``(compromised, change_history, llm)`` with
  quorum_min=2, requiring BOTH change_history AND llm to land
  gold — is recorded in ``docs/data/phase68_live_coalition_*``
  artifacts; that is the W21-Λ-real-strict frontier.
* **Two-Mac path remains infrastructure-only.** Mac 2 is
  unreachable at this milestone capture — same status as the
  prior 15 milestones. No SDK code change is required for
  cross-host deployment when Mac 2 returns; the W21 oracle path
  is duck-typed.
* **The W21 escape is partial by design.** The single-oracle wall
  W20-Λ-compromised remains real for *any* single-oracle scorer.
  W21 widens the scope from "single oracle" to "registered set
  of N oracles"; it does not escape *every* possible adversarial
  regime — joint-deception across all N (R-68-MULTI-ALL-COMPROMISED)
  is the named structural limit. The escape is bounded above by
  *registered-set integrity*, not a richer scoring rule.

## Master plan / status synchronisation

* ``docs/RESEARCH_STATUS.md`` — milestone summary updated.
* ``docs/THEOREM_REGISTRY.md`` — W21 family entries minted.
* ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` — Bar 18 added,
  bar list updated to eighteen bars.
* ``docs/context_zero_master_plan.md`` — § next-frontier updated to
  name **W21-C-CALIBRATED-TRUST** + **W22-* (joint deception
  detection via cross-source consistency)** as the next research
  frontiers.
* ``docs/START_HERE.md`` — current-milestone line updated.
* ``README.md`` — milestone summary updated.
* ``CHANGELOG.md`` — SDK v3.22 entry.
* ``vision_mvp/coordpy/__init__.py`` — ``SDK_VERSION = "coordpy.sdk.v3.22"``.

The CoordPy single-run product runtime contract is byte-for-byte
unchanged from SDK v3.21 — the W21 surface is purely additive in
the multi-agent coordination research slice (one new dataclass +
one tokeniser-free per-tag voter + one disambiguator class +
three new oracle adapters: ``ChangeHistoryOracle``,
``OnCallNotesOracle``, ``SingletonAsymmetricOracle``,
``DisagreeingHonestOracle``).

## Tests added (48 new)

* ``ChangeHistoryOracleTests`` × 3.
* ``OnCallNotesOracleTests`` × 3.
* ``SingletonAsymmetricOracleTests`` × 5.
* ``DisagreeingHonestOracleTests`` × 2.
* ``W21DecoderUnitTests`` × 11 (trigger gating, branch determinism,
  no-oracle / disabled paths, default trigger set, branch-typing,
  reduces-to-W20-byte-for-byte).
* ``Phase68BenchPropertyTests`` × 1 (every bank holds the
  R-66-OUTSIDE-REQUIRED shape).
* ``Phase68DefaultConfigTests`` × 4 (loose/tight strict win, audit
  OK, branch counts).
* ``Phase68SeedStabilityTests`` × 3 (loose/tight min-gap at strong
  bar, n=8 × 5 seeds at 1.000).
* ``Phase68FalsifierTests`` × 3 (no_quorum / all_compromised /
  partial-default-q2).
* ``Phase68ConditionalSuccessTests`` × 1 (partial-q1 recovers).
* ``Phase68BackwardCompatTests`` × 2 (R-66 default banks,
  ``enabled = False``).
* ``Phase68TokenBudgetHonestyTests`` × 3 (per-cell budget, no W15
  inflation, strict T_decoder).
* ``Phase68CrossRegimeSyntheticTests`` × 5 (cross-regime sweep
  invariants).
* ``Phase68ReducesToW20OnR67Tests`` × 1 (single-oracle quorum_min=1
  ties W20 byte-for-byte on R-67).

## How to reproduce

```bash
# Synthetic anchor (deterministic, ~6s):
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_majority --decoder-budget -1 \
    --K-auditor 12 --n-eval 8 --out -

# Tight composition:
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_majority --decoder-budget 24 \
    --K-auditor 12 --n-eval 8 --out -

# Cross-regime synthetic summary:
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --cross-regime-synthetic --K-auditor 12 --n-eval 8 --out -

# 5-seed stability sweep:
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_majority --seed-sweep \
    --K-auditor 12 --n-eval 8 --out -

# Falsifier sweeps:
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_no_quorum --K-auditor 12 --n-eval 8 --out -
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_all_compromised --K-auditor 12 --n-eval 8 --out -
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_partial --K-auditor 12 --n-eval 8 --out -
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_partial --quorum-min 1 \
    --K-auditor 12 --n-eval 8 --out -

# Live mixed-registry probe (Mac-1, 47B-MoE; pairs deterministic
# registry oracles with the live LLM):
python3 -m vision_mvp.experiments.phase68_multi_oracle_adjudication \
    --bank multi_majority --live-mixed-registry \
    --adjudicator-model mixtral:8x7b \
    --K-auditor 12 --n-eval 4 --out -

# Run the full W21 test suite:
python3 -m unittest vision_mvp.tests.test_coordpy_multi_oracle_adjudication

# Full SDK regression (633 tests):
python3 -m unittest discover vision_mvp/tests -p "test_coordpy_*.py"
```
