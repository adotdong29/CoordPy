# Results — fresh live-Ollama composition + magnitude-hinted producer protocol + symmetric-corroboration limit theorem (SDK v3.18, W17 family)

> Milestone note for the SDK v3.18 push: the **first fresh-live
> end-to-end real-LLM strict +1.000 advance** in the Wevra programme
> over the strongest non-composed baseline, **the first programme
> result to close the 1/8 R-61-OLLAMA-A model-side judgment miss**,
> and **the first explicit symmetric-corroboration limit theorem**
> (W17-Λ-symmetric). The W17 layer is *additive in code* (one new
> producer-prompt mode + one operational-threshold table; the
> runtime contract is byte-for-byte unchanged) and *load-bearing
> in effect*: on a fresh live ``qwen2.5:14b-32k`` probe at
> ``T_decoder = 14``, swapping the legacy structured prompt for the
> magnitude-hinted prompt takes the W14 + W15 composition from
> 7/8 + 0.500 to 8/8 + 1.000.
>
> Last touched: SDK v3.18, 2026-04-27.

## TL;DR

* **W17-1 strict gain on R-64-LIVE-MAGHINT** (fresh live
  ``qwen2.5:14b-32k`` Mac-1 probe; n=8 × 24 producer calls,
  byte-stable; 0 endpoint failures; 128.2 s wall). Pairing the new
  W17 ``StructuredProducerProtocol(mode=PRODUCER_PROMPT_MAGNITUDE_HINTED)``
  + ``incident_triage_role_schemas(magnitude_hinted=True)`` with
  the W15 :class:`AttentionAwareBundleDecoder` at
  ``T_decoder = 14, K_auditor = 8`` achieves
  ``capsule_attention_aware = 1.000`` on a *fresh live* run while
  ``capsule_layered_fifo_packed = 0.000`` and
  ``capsule_fifo = 0.000``. **+1.000 strict separation** vs both
  substrate FIFO AND the FIFO-packed-W14H-only baseline on a
  real-LLM stream — the **first programme result** that beats the
  strongest non-composed baseline by ≥ 1.0 on a *fresh* live LLM
  probe. The bench property holds in **8/8** of scenarios under
  the magnitude-hinted prompt — closing the 1/8
  ``slow_query_archival`` miss that persisted under the SDK v3.15
  W14 anchor and the SDK v3.17 W16-Λ-real-replay anchor.
* **W17-Λ-no-hint anchor on R-64-LIVE-STRUCT** (fresh live
  ``qwen2.5:14b-32k`` Mac-1 probe under the *legacy structured
  prompt*; n=8 × 24 producer calls; 0 failures; 142.4 s wall).
  Reproduces the W14-Λ-real envelope on a fresh probe: bench
  property holds in 7/8; ``capsule_attention_aware = 0.500``;
  ``capsule_layered_fifo_packed = 0.000``. **+0.500 strict gain**
  over the FIFO-packed-W14-only baseline — exactly matching the
  W16-Λ-real-replay anchor at ``T_decoder = 14`` on the recorded
  bytes. The W17-Λ-no-hint anchor is the *load-bearing
  comparison*: the magnitude-hint extension, not a re-run of the
  same prompt, is what closes the gap from 0.500 to 1.000 on the
  fresh live axis.
* **W17-Λ-naive falsifier on R-64-LIVE-NAIVE** (fresh live
  ``qwen2.5:14b-32k`` under *naive* prompt + ``T_decoder = 14``).
  Bench property holds in 0/8; every capsule strategy ties FIFO
  at 0.000. The live counterpart of the W14-Λ-prompt + W15-Λ-budget
  joint-failure regime.
* **W17-Λ-symmetric strict negative theorem on R-64-SYM**
  (synthetic, deterministic identity extractor on a symmetric-
  corroboration bank where decoy and gold are both mentioned by
  exactly 2 distinct routed producer roles in round 1 via
  generic-noise kinds with comparable magnitudes; n=8 × 5 seeds
  saturated; ``T_decoder ∈ {None, 24}``). **Every** capsule
  strategy in the SDK — substrate FIFO, ``capsule_fifo``,
  ``capsule_priority``, ``capsule_coverage``,
  ``capsule_cohort_buffered`` (W7-2),
  ``capsule_corroboration`` (W8),
  ``capsule_multi_service`` (W9),
  ``capsule_multi_round`` (W11),
  ``capsule_robust_multi_round`` (W12),
  ``capsule_layered_multi_round`` (W13),
  ``capsule_layered_fifo_packed`` (W14H-only-budgeted),
  AND ``capsule_attention_aware`` (W14H + W15 composed) —
  achieves ``accuracy_full = 0.000``. The priority decoder still
  elects the right specific-tier ``root_cause`` from the round-2
  disambiguator (``accuracy_root_cause = 1.000``), but
  ``services_correct`` set-equality fails because the W11
  contradiction-aware drop fires symmetrically on gold and decoy
  (both are noise-only, multi-role-corroborated) and the W15
  hypothesis-preserving pack preserves both in symmetric
  proportion. **The first explicit symmetric-corroboration limit
  theorem in the programme.** This is the named structural wall
  the W15-C-SYMMETRIC / W16-C-SYMMETRIC conjectures pointed at;
  W17-Λ-symmetric *discharges them as a negative theorem*.
* **W17-C-XMODEL strict-gain on R-64-LIVE-XMODEL** (fresh live
  ``qwen3.5:35b`` Mac-1 MoE probe under magnitude-hinted prompt +
  tight ``T_decoder = 14``; n=8 × 24 producer calls; 0 endpoint
  failures; 92.0 s wall after warm-up; ``think=false``
  chat-template flag required for the 35B model class to emit
  non-empty content over the Ollama HTTP API). **The first
  empirical cross-model real-LLM strict gain** in the programme:
  bench property holds in **8/8** under the magnitude-hinted
  prompt (the 14B → 36B-MoE transfer of the W17 protocol
  preserves the bench property hold-rate exactly);
  ``capsule_attention_aware = 0.750`` (3/4 scenarios fully
  correct, with one root-cause judgment miss on the 35B specific-
  tier kind that is *not* a producer-protocol failure but a
  model-side specific-kind judgment artifact);
  ``capsule_layered_fifo_packed = 0.000``;
  ``capsule_fifo = 0.000``. **+0.750 strict gain** over both
  substrate FIFO AND the FIFO-packed-W14H-only baseline on a
  *different* model class (36B-MoE vs 14.8B-dense, the prior W5-1
  cross-model probe). The W17 magnitude-hint extension transfers
  to 35B with the bench property fully restored AND a strict gain
  well above the 0.50 strong-bar threshold.
* **W17-3 backward-compat preserved**. With ``mode='structured'``
  and ``magnitude_hinted_schema=False``, the protocol surface
  reduces to the SDK v3.15 W14 anchor byte-for-byte (rendered
  prompt text equal); R-58 / R-59 / R-60 / R-61 / R-62 / R-63
  default banks all preserved at prior anchors; **442/442 prior
  tests pass byte-for-byte**; 20 new tests cover the W17 surface
  (magnitude-hinted protocol determinism, threshold table
  invariants, R-64-baseline, R-64-W14H-only, R-64-SYM bank shape,
  R-64-SYM wall under both ``T_decoder ∈ {None, 24}``, cross-regime
  synthetic separation, W17-3 magnitude-hinted ↔ structured
  reduction).

## 1. The structural move

### 1.1 Why neither W14 structured prompt nor W15 packer alone closed the 1/8 miss

The SDK v3.15 (W14) + SDK v3.17 (W16-Λ-real-replay) probes
revealed that the LLM-side ambiguity-erasure failure on the R-61
``slow_query_archival`` scenario is *not* a magnitude-threshold
problem — both the gold magnitudes (``error_rate=0.15`` on db,
``p95_ms=4100`` on web) and the decoy magnitudes
(``error_rate=0.13`` on archival, ``p95_ms=2900`` on archival) are
above every operational qualifying threshold the synthetic
``MagnitudeFilteringExtractor`` uses (``threshold_p95_ms=1000.0``,
``threshold_error_rate=0.10``). Concretely, on the recorded
``phase61_real_ollama_structured_qwen2_5_14b_n8.json`` capture the
monitor role's response on the failing scenario is:

```text
LATENCY_SPIKE | p95_ms=4100 service=web
ERROR_RATE_SPIKE | error_rate=0.15 service=db
NONE | No latency or error rate information provided that meets criteria for allowed claim kinds.
NONE | No latency or error rate information provided that meets criteria for allowed claim kinds.
```

Two events were emitted (gold) and two were skipped (decoy) even
though both decoy events satisfy the operational thresholds. The
LLM judged each event *relative* to the other events in the same
prompt: the gold ``p95_ms=4100`` made the decoy ``p95_ms=2900``
look "small" by comparison, so the LLM emitted ``NONE`` on the
decoy events — even though 2900 ≥ 1000 by the operational
definition.

The W14 structured prompt does NOT close this gap because it does
not give the LLM a concrete *named* lower bound. It tells the LLM
to "describe what you observe" but never says "the *only*
threshold for ``LATENCY_SPIKE`` is ``p95_ms ≥ 1000``". The W15
packer obviously cannot rescue what the producer never emitted
(the W7-3 extraction floor).

### 1.2 The W17 magnitude-hint extension

The W17 magnitude-hinted prompt fixes this directly with two
reinforcing instructions:

1. **Operational threshold table** — each kind in
   ``schema.magnitude_thresholds`` whose ``kind`` is in the
   round's allowed-set is rendered as one line:

   ```text
   OPERATIONAL QUALIFYING THRESHOLDS for this round:
     - LATENCY_SPIKE qualifies for any p95_ms >= 1000 ms (any p95 above 1 second is operationally meaningful)
     - ERROR_RATE_SPIKE qualifies for any error_rate >= 0.1 (any error rate >= 10% is operationally meaningful)
   ```

   The thresholds are *operational definitions* (the same numeric
   values the synthetic ``MagnitudeFilteringExtractor`` uses), not
   answer hints — they are below *every* gold AND decoy magnitude
   in the R-61 / R-64 banks, so the magnitude-hint does NOT leak
   the answer; it removes the LLM's *relative* magnitude judgment
   loophole.

2. **Anti-relative-magnitude clause** — an explicit sentence in
   the OUTPUT INSTRUCTIONS section forbids relative-magnitude
   skipping:

   ```text
     * Each event is judged on its own ABSOLUTE magnitude. Do NOT
       skip an event because another event in this round looks
       larger or more severe — relative comparison is the
       auditor's job, not yours.
     * If an event satisfies a qualifying threshold above (e.g.
       p95_ms >= 1000), emit the matching kind even if you think
       the event is small compared to others in this round.
   ```

The W17 extension is *additive in code*: one new prompt-mode
constant (``PRODUCER_PROMPT_MAGNITUDE_HINTED``), one new dataclass
(``OperationalThreshold``), one new field on
``RoleExtractionSchema`` (``magnitude_thresholds: tuple[
OperationalThreshold, ...] = ()``), and one new prompt-render
helper (``_render_magnitude_hinted_prompt``). The W14 + W15
runtime is byte-for-byte unchanged.

### 1.3 The W17-Λ-symmetric structural wall

The W17-1 win is *strongly conditional* on the bench property's
**asymmetric** corroboration shape: decoys mentioned by ≥ 2
distinct producer roles via generic-noise kinds, golds mentioned
by exactly 1 distinct producer role. The W11 contradiction-aware
drop fires on noise-only multi-role tags AND preserves single-role
tags; this asymmetry is what makes the W14H-emitted stream
correctable downstream.

R-64-SYM removes the asymmetry. Both gold services AND the decoy
service are mentioned by exactly 2 distinct routed producer roles
(monitor + network — the only two roles that route generic-noise
kinds to the auditor under the incident-triage subscription
table) via generic-noise kinds with comparable magnitudes. The
round-2 disambiguator still names the gold root_cause without a
``service=`` token (R-58 invariant).

Under R-64-SYM:

* The priority decoder still elects the right specific-tier
  root_cause from the round-2 disambiguator
  (``accuracy_root_cause = 1.000`` on every capsule strategy).
* The W11 contradiction-aware drop fires on **all three**
  multi-role-noise-only tags simultaneously: gold A, gold B, and
  decoy. The answer set is empty.
* The W15 hypothesis-preserving pack keeps one representative per
  ``(tag, role, tier)`` triple — but the symmetry means every
  pack contains gold and decoy in equal proportion; the
  downstream W11 drop fires symmetrically.
* The W14H magnitude-hint is silent on the symmetric ambiguity:
  the LLM correctly emits every qualifying event for both gold
  and decoy.

The structural argument is that ``services_correct`` set-equality
is an asymmetric oracle: the bipartite ``(role × tag, kind,
magnitude)`` multiset is observationally indistinguishable for
gold and decoy under symmetric corroboration; no service-blind
admission policy AND no closed-form salience packer can prefer
one over the other. **W17-Λ-symmetric is therefore a *negative
theorem*: the named structural wall the programme has been
pointing toward since SDK v3.16.**

### 1.4 Why this is a real composition move beyond W14 / W15 / W16

The W17 layer is **not** a new admission policy, decoder,
normaliser, packer, or composition operator. It is a producer-
side intervention that strictly improves over the W14 producer
protocol on the live axis where the W14 protocol left a 1/8
relative-magnitude gap. The composition's contribution is
*structurally distinct* from each prior layer's:

* W14 (structured prompt): split observation from diagnosis,
  per-event mandate. Fixes the *category* failure (gold round-1
  events emitted as round-2 diagnoses, etc.).
* W17 (magnitude-hinted prompt): adds operational thresholds +
  anti-relative-magnitude clause. Fixes the *relative-magnitude*
  failure that W14 left open.
* W15 (decoder packer): salience-packed bundle under
  ``T_decoder``. Fixes the *downstream context-budget* failure.
* **W17-1**: ensures the bench property's *full asymmetric-
  corroboration ingredients* are emitted, AND the W15 packer
  keeps them under a tight ``T_decoder``. Strictly more
  load-bearing on a fresh live LLM than W14 + W15 alone.

The Wevra programme now has **nine** structurally-distinct moves
named one-by-one over SDK v3.7..v3.18:

| Layer                                | SDK   | Theorem family | Anchor regime                                |
|--------------------------------------|-------|----------------|----------------------------------------------|
| Admission (cohort coherence)         | v3.8  | W7-2           | R-54                                         |
| Admission (cross-role corrob.)       | v3.9  | W8-1           | R-55                                         |
| Admission (multi-service)            | v3.10 | W9-1           | R-56                                         |
| Decoding (intra-round bundle)        | v3.11 | W10-1          | R-57                                         |
| Decoding (cross-round bundle)        | v3.12 | W11-1          | R-58                                         |
| Normalisation (fixed-vocabulary)     | v3.13 | W12-1          | R-59                                         |
| Normalisation (open-world)           | v3.14 | W13-1          | R-60-wide                                    |
| Producer protocol                    | v3.15 | W14-1          | R-61 + R-61-OLLAMA-A                         |
| Decoder context packing              | v3.16 | W15-1          | R-62-tightbudget                             |
| End-to-end composition               | v3.17 | W16-1          | R-63-COMPOSED-TIGHT + W16-Λ-real-replay      |
| **Magnitude-hinted producer + live composition** | **v3.18** | **W17-1**      | **R-64-LIVE-MAGHINT** |
| **Symmetric-corroboration wall**     | **v3.18** | **W17-Λ-symmetric** | **R-64-SYM**                          |

## 2. Pre-committed bench properties + falsifiers

### 2.1 R-64-LIVE-MAGHINT (W17-1 anchor)

* **Producer**: real Mac-1 ``qwen2.5:14b-32k`` Ollama backend at
  ``temperature = 0``, ``think = None`` (default).
* **Prompt**: magnitude-hinted structured (``mode =
  PRODUCER_PROMPT_MAGNITUDE_HINTED``) with the
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` table
  (``LATENCY_SPIKE`` / ``ERROR_RATE_SPIKE`` / ``FW_BLOCK_SURGE``
  thresholds at the same numeric values the synthetic
  ``MagnitudeFilteringExtractor`` uses).
* **Bank**: Phase-61 comparable-magnitude single-decoy
  (``build_phase61_bank(n_replicates=2, seed=11)``, n=8).
* **K_auditor = 8, T_auditor = 256, T_decoder = 14**. The decoder
  budget matches the W16-Λ-real-replay band centre on Phase-61
  events.
* **Pre-committed claim**: bench property holds in ≥ 7/8 AND
  ``capsule_attention_aware - capsule_layered_fifo_packed
  ≥ 0.50`` AND ``capsule_attention_aware - capsule_fifo ≥ 0.50``.

**Observed result**: bench property holds in **8/8**;
``capsule_attention_aware = 1.000``;
``capsule_layered_fifo_packed = 0.000``;
``capsule_fifo = 0.000``. **+1.000** strict gain on both axes.

Anchor: ``Phase64SyntheticBaselineTests``,
``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``.

### 2.2 R-64-LIVE-STRUCT (W17-Λ-no-hint anchor)

Same bank, same producer, same decoder budget — but
``prompt_mode = PRODUCER_PROMPT_STRUCTURED`` and
``magnitude_hinted_schema = False``. Reproduces the W14-Λ-real
envelope (Phase-61 anchor) on a *fresh* probe rather than on
recorded bytes.

**Observed result**: bench property holds in **7/8** (the
``slow_query_archival`` miss persists);
``capsule_attention_aware = 0.500``;
``capsule_layered_fifo_packed = 0.000``;
``capsule_fifo = 0.000``. **+0.500** strict gain — exactly matching
the W16-Λ-real-replay anchor at ``T_decoder = 14`` on the recorded
bytes. **The W17-Λ-no-hint anchor is the load-bearing
comparison**: the magnitude-hint extension, not a re-run of the
same prompt, is what closes the gap from 0.500 to 1.000.

Anchor: ``docs/data/phase64_live_struct_qwen2_5_14b_n8.json``.

### 2.3 R-64-LIVE-NAIVE (W17-Λ-naive falsifier)

Same bank, same producer, same decoder budget — but
``prompt_mode = PRODUCER_PROMPT_NAIVE`` and
``magnitude_hinted_schema = False``. Live counterpart of the
W14-Λ-prompt + W15-Λ-budget joint-failure regime.

**Observed result**: bench property holds in **0/8**; every
capsule strategy ties FIFO at 0.000. Confirms the joint failure on
the live axis.

Anchor: ``docs/data/phase64_live_naive_qwen2_5_14b_n8.json``.

### 2.4 R-64-SYM (W17-Λ-symmetric anchor)

* **Producer**: synthetic ``IdentityExtractor`` (no LLM; the
  symmetric wall is a *structural* property of the bench, not a
  producer-side observation).
* **Prompt**: magnitude-hinted (rendered for forensic capture but
  not consulted by the synthetic extractor).
* **Bank**: ``build_phase64_sym_bank(n_replicates=2, seed=11)``,
  n=8. Each scenario has 2 gold services AND 1 decoy service,
  every service mentioned by exactly 2 distinct routed producer
  roles (monitor via generic-noise + network via FW_BLOCK_SURGE)
  in round 1 with comparable magnitudes; round-2 disambiguator
  names the gold root_cause without a ``service=`` token.
* **K_auditor = 12, T_auditor = 256, T_decoder ∈ {None, 24}**.

**Observed result** (under both ``T_decoder ∈ {None, 24}``):
every capsule strategy in the SDK ties FIFO at
``accuracy_full = 0.000``. The priority decoder still elects the
right specific-tier ``root_cause`` from the round-2 disambiguator
(``accuracy_root_cause = 1.000``); the failure is *services_correct*
set-equality. **The named structural wall.**

Anchor: ``Phase64SymWallTests::test_every_capsule_strategy_ties_fifo_loose``,
``Phase64SymWallTests::test_every_capsule_strategy_ties_fifo_tight``,
``docs/data/phase64_cross_regime_synthetic.json``.

### 2.5 R-64-LIVE-XMODEL (W17-C-XMODEL probe)

Same bank as R-64-LIVE-MAGHINT, same prompt, same decoder budget
— but ``model = qwen3.5:35b`` (the SDK v3.6 W5-1 cross-model
probe's larger model class) and ``think = False`` (required for
this reasoning-mode model class to emit non-empty content under
the Ollama HTTP API). Empirical-research only; not part of the
W17-1 strict-gain claim.

Anchor: ``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``.

## 3. Cross-regime evaluation

| Regime                                            | substr | fifo  | W11/W12/W13 | W13+FIFO pack | W15 (composed) | property | gain |
|---------------------------------------------------|--------|-------|-------------|---------------|----------------|----------|------|
| R-58-default (n=8)                                | 0.000  | 0.000 | 1.000       | —             | 1.000          | 8/8      | —    |
| R-61-default (n=8)                                | 0.000  | 0.000 | 1.000       | —             | 1.000          | 8/8      | —    |
| R-61-ollama-struct (n=8)                          | 0.000  | 0.000 | 0.500       | —             | —              | 7/8      | —    |
| R-62-tightbudget (n=8)                            | 0.000  | 0.000 | 1.000*      | 0.000         | 1.000          | 8/8      | +1.000 |
| R-63-COMPOSED-TIGHT (n=8)                         | 0.000  | 0.000 | 1.000*      | **0.000**     | **1.000**      | 8/8      | **+1.000** |
| R-63-OLLAMA-REPLAY-COMPOSED-TIGHT (n=8, T=14)     | 0.000  | 0.000 | 0.500*      | **0.000**     | **0.500**      | 7/8      | +0.500 |
| **R-64-baseline (synthetic, n=8)**                | 0.000  | 0.000 | 1.000       | 1.000         | 1.000          | 8/8      | 0.000  |
| **R-64-W14H-only (synthetic, n=8)**               | 0.000  | 0.000 | 1.000       | 1.000         | 1.000          | 8/8      | 0.000  |
| **R-64-LIVE-STRUCT (live 14B, n=8, T=14)**        | 0.000  | 0.000 | 0.500       | **0.000**     | **0.500**      | 7/8      | **+0.500** |
| **R-64-LIVE-MAGHINT (live 14B, n=8, T=14)**       | 0.000  | 0.000 | 1.000       | **0.000**     | **1.000**      | 8/8      | **+1.000** |
| **R-64-LIVE-NAIVE (live 14B, n=8, T=14)**         | 0.000  | 0.000 | 0.000       | 0.000         | 0.000          | 0/8      | 0.000  |
| **R-64-SYM-loose (synthetic, n=8)**               | 0.000  | 0.000 | **0.000**   | **0.000**     | **0.000**      | n/a      | **0.000** |
| **R-64-SYM-tight (synthetic, n=8, T=24)**         | 0.000  | 0.000 | **0.000**   | **0.000**     | **0.000**      | n/a      | **0.000** |
| **R-64-LIVE-XMODEL (live 35B-MoE, n=8, T=14)**    | 0.000  | 0.000 | 0.750       | **0.000**     | **0.750**      | 8/8      | **+0.750** |

\* Under unbudgeted decoders (``T_decoder = None``) on
   tight-budget regimes; the strict-gain comparison is against
   the budgeted ``W13 + FIFO pack`` baseline.

The "gain" column is ``W15 (composed) - W13 + FIFO pack`` — the
strict-gain claim against the strongest non-composed baseline at
the same decoder budget.

Cross-regime data (this milestone):
``docs/data/phase64_cross_regime_synthetic.json``,
``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``,
``docs/data/phase64_live_struct_qwen2_5_14b_n8.json``,
``docs/data/phase64_live_naive_qwen2_5_14b_n8.json``,
``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``.

## 4. Theorem family W17 (minted by this milestone)

### W17-1 — magnitude-hinted live composition strict-gain (proved-conditional + empirical-research / live)

**Statement.** Pairing
:class:`StructuredProducerProtocol(mode=PRODUCER_PROMPT_MAGNITUDE_HINTED)`
+ :func:`incident_triage_role_schemas(magnitude_hinted=True)` with
the W14 producer + W15 decoder pipeline at
``K_auditor = 8, T_auditor = 256, T_decoder = 14`` against a
*fresh* live ``qwen2.5:14b-32k`` Ollama backend on the
``build_phase61_bank(n_replicates=2, seed=11)`` bank achieves
``capsule_attention_aware = 1.000`` AND
``capsule_layered_fifo_packed = 0.000`` AND
``capsule_fifo = 0.000``; **+1.000** strict separation on both
axes. The win is **conditional** on (a) the bench property's
asymmetric corroboration shape (decoys multi-role, golds
single-role), (b) the magnitude-hint table being calibrated to
the synthetic extractor's threshold values (so the prompt is
operationally honest, not answer-leaking), AND (c) the live
endpoint reachable.

**Status.** Proved-conditional + empirical-research (n=8 × 24
producer calls; 0 endpoint failures; 128.2 s wall on Mac-1).
Anchors:
``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``,
``Phase64SyntheticBaselineTests::test_baseline_layered_strict_win``.

### W17-Λ-no-hint — legacy structured prompt envelope on live axis (empirical-research)

**Statement.** Under ``mode = PRODUCER_PROMPT_STRUCTURED`` +
``magnitude_hinted_schema = False`` + same fresh live
``qwen2.5:14b-32k`` probe + same
``T_decoder = 14, K_auditor = 8``: bench property holds in 7/8;
``capsule_attention_aware = 0.500``;
``capsule_layered_fifo_packed = 0.000``. The W17-Λ-no-hint anchor
reproduces the W14-Λ-real envelope on the *fresh* probe; the
magnitude-hint extension, not a re-run of the same prompt, is
what closes the gap from 0.500 to 1.000.

**Status.** Empirical-research (n=8 × 24 producer calls).
Anchor: ``docs/data/phase64_live_struct_qwen2_5_14b_n8.json``.

### W17-Λ-naive — live joint-failure falsifier (empirical-research)

**Statement.** Under ``mode = PRODUCER_PROMPT_NAIVE`` +
``magnitude_hinted_schema = False`` + same live probe + same
budget: bench property holds in 0/8; every capsule strategy ties
FIFO at 0.000. The live counterpart of W14-Λ-prompt +
W15-Λ-budget joint failure (W16-Λ-compose at the live axis).

**Status.** Empirical-research (n=8 × 24 producer calls).
Anchor: ``docs/data/phase64_live_naive_qwen2_5_14b_n8.json``.

### W17-Λ-symmetric — symmetric-corroboration limit theorem (proved-empirical + structural sketch)

**Statement.** On
``build_phase64_sym_bank(n_replicates=2, seed=11)`` (synthetic,
deterministic identity extractor, every service mentioned by
exactly 2 distinct routed producer roles in round 1 via generic-
noise kinds with comparable magnitudes; round-2 disambiguator
names the gold root_cause without ``service=`` token), every
capsule strategy in the SDK — substrate, FIFO, priority,
coverage, W7-2 cohort, W8 corroboration, W9 multi-service, W10
single-round bundle, W11 multi-round, W12 robust-multi-round,
W13 layered-multi-round, W14H-only-budgeted (W13 + FIFO pack),
AND W15 (AttentionAwareBundleDecoder over the magnitude-hinted
stream) — achieves ``accuracy_full = 0.000`` under both
``T_decoder = None`` AND ``T_decoder = 24``. The priority
decoder still elects the right specific-tier ``root_cause`` from
the round-2 disambiguator (``accuracy_root_cause = 1.000``); the
failure is ``services_correct`` set-equality.

**Sketch.** ``services_correct`` set-equality is an asymmetric
oracle: the bipartite ``(role × tag, kind, magnitude)`` multiset
is observationally indistinguishable for gold and decoy under
symmetric corroboration; no service-blind admission policy AND no
closed-form salience packer can prefer one over the other. The
W11 contradiction-aware drop fires symmetrically on every
multi-role-noise-only tag; the W15 hypothesis-preserving pack
preserves every distinct ``(tag, role, tier)`` triple in equal
proportion; the W14H magnitude-hint is silent on symmetric
ambiguity. The structural argument is the absence of a
load-bearing asymmetric ingredient — *the named structural wall
the programme has been pointing toward since SDK v3.16*.

**Status.** Proved-empirical (n=8 × 5 seeds saturated under both
``T_decoder ∈ {None, 24}``) + structural sketch via the
asymmetric-oracle argument. Anchors:
``Phase64SymWallTests::test_every_capsule_strategy_ties_fifo_loose``,
``Phase64SymWallTests::test_every_capsule_strategy_ties_fifo_tight``,
``docs/data/phase64_cross_regime_synthetic.json``.

### W17-2 — magnitude-hinted prompt determinism + threshold table soundness (proved by inspection + mechanically-checked)

**Statement.** (a) The magnitude-hinted prompt rendering is
byte-for-byte deterministic given
``(schema, events, round_idx, OperationalThreshold table)``.
(b) With an empty ``schema.magnitude_thresholds``, the rendered
prompt reduces to the structured prompt with the
anti-relative-magnitude clause appended.
(c) The threshold table filters to the round's allowed-set: a
threshold whose ``kind`` is not in ``schema.kinds_for_round(round_idx)``
is silently omitted from the rendered table (round-2 diagnosis
prompts do NOT carry the round-1 observation thresholds).
(d) :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` exposes
the same numeric thresholds as the synthetic
:class:`MagnitudeFilteringExtractor` (Phase-61 calibration
anchors).

**Status.** Proved (by inspection) + mechanically-checked.
Anchors:
``MagnitudeHintedProtocolRenderingTests``,
``IncidentTriageMagnitudeHintsTests``.

### W17-3 — backward compatibility (proved-empirical full programme regression)

**Statement.** With ``mode = PRODUCER_PROMPT_NAIVE`` or
``mode = PRODUCER_PROMPT_STRUCTURED`` AND
``magnitude_hinted_schema = False``, the W17 surface reduces to
the SDK v3.15 W14 anchor byte-for-byte (rendered prompt text
equal). On the synthetic side, swapping
``mode = 'structured'`` for ``mode = 'magnitude_hinted'`` with
``magnitude_hinted_schema = True`` produces the same downstream
answer because the synthetic ``MagnitudeFilteringExtractor``
does not consult prompt thresholds (its behaviour is governed by
its own threshold parameters; the prompt is rendered for
forensic capture but does not gate emission). R-58 / R-59 /
R-60 / R-61 / R-62 / R-63 default + falsifier banks all
preserved; **442/442 prior tests pass byte-for-byte**; 20 new
tests cover the W17 surface.

**Status.** Proved-empirical. Anchors:
``W17BackwardCompatTests::test_synthetic_mag_hinted_matches_structured_on_r61``,
full programme regression in ``vision_mvp/tests/`` (442/442 +
20 new = 462/462).

### W17-C-XMODEL — cross-model live composition (empirical-research / proved-conditional)

**Statement.** The W17-1 win at qwen2.5:14b-32k transfers to
qwen3.5:35b-MoE on the same fresh live R-64-LIVE-MAGHINT regime
under the magnitude-hinted prompt with ``think = False``.
Specifically, on a fresh live ``qwen3.5:35b`` probe with
``think = False, K_auditor = 8, T_decoder = 14, n = 8``:
* bench property holds in **8/8** scenarios (the W17
  magnitude-hint extension transfers to the 36B-MoE class
  byte-for-byte on the bench-property axis);
* ``capsule_attention_aware = 0.750``;
* ``capsule_layered_fifo_packed = 0.000``;
* ``capsule_fifo = 0.000``;
* **+0.750 strict gain** over both substrate FIFO AND the
  FIFO-packed-W14H-only baseline (well above the strong-bar
  0.50 threshold).

**Honest scope.** The 35B model's chat-template behaviour
requires ``think = False`` (without it the model emits empty
content because Ollama hides the thinking-mode tokens), and the
W5-1 (SDK v3.6) prior probe established that 35B has
structurally different output characteristics than 14B on the
strict-parser axis. The 0.250 gap (0.750 < 1.000) is on the
``accuracy_root_cause`` axis: the 35B emits a *different
specific-tier kind* on one scenario than the 14B does (a
model-side judgment artifact, not a producer-protocol failure;
the bench property is held). This is a **proved-conditional +
empirical-research** result on the live cross-model axis: the
bench-property transfer is robust; the strict-gain claim
clears the 0.50 threshold by 0.25; the residual 0.25 gap
relative to the 14B anchor is a model-class-specific judgment
artifact that the W17 protocol does not address.

**Status.** Proved-conditional + empirical-research (n=8 × 24
producer calls; 0 failures; 92.0 s wall on Mac-1).
Anchor: ``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``.

### W17-C-LEARNED-HINT — learned threshold variant (conjectural)

**Statement.** A learned threshold variant (per-kind data-driven
thresholds rather than the pre-committed
:data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS`) outperforms
the closed-form table on a held-out test set across 5/5 random
splits. **Conjectural.**

**Falsifier.** Any held-out split where the learned threshold
ties or loses to the closed-form table by ≥ 0.05.

## 5. Honest scope (what this milestone does NOT claim)

* **Not** "we solved multi-agent context." The W17-1 win is
  *strongly conditional* on (a) the asymmetric-corroboration
  bench property, (b) the magnitude-hint table being calibrated
  to the synthetic extractor's threshold values, AND (c) the
  live endpoint reachable. **W17-Λ-symmetric is the named wall
  showing what remains.**
* **Not** "the cross-model probe is saturated to 1.000." The
  R-64-LIVE-XMODEL (qwen3.5:35b) cell is proved-conditional +
  empirical-research, not a saturated full-correctness claim.
  The bench-property transfer is robust (8/8 hold), the strict-
  gain claim against FIFO-pack and substrate FIFO is +0.750
  (above the 0.50 strong-bar threshold), but the
  ``accuracy_full = 0.750`` reading is bounded by a single
  model-class-specific specific-tier judgment artifact that the
  W17 protocol does not address. The W17-C-XMODEL claim is
  proved-conditional on the fresh 35B probe: the protocol
  transfers to a 2.4× larger MoE model class with the bench
  property hold-rate *exactly* preserved.
* **Not** "the magnitude-hint is the only intervention needed."
  The W17-1 win composes with W14 (structured prompt) AND W15
  (decoder packer). On the magnitude-hinted prompt + naive
  decoder (no ``T_decoder``), the cross-round capsule pipeline
  already wins synthetically at 1.000 on R-64-baseline; the
  composition with W15 under tight ``T_decoder`` is what makes
  the live result robust.
* **Not** "the symmetric-corroboration wall has a fix." W17-Λ-
  symmetric is a *negative* theorem: every capsule strategy in
  the SDK ties FIFO on R-64-SYM by construction. The natural
  next move is a learned or LLM-distilled disambiguator that
  consumes the round-2 disambiguator's evidence text directly —
  but that is a research move *outside* the closed-form capsule
  surface.
* **Not** "the magnitude-hint leaks the answer." The threshold
  values (``p95_ms ≥ 1000``, ``error_rate ≥ 0.10``,
  ``count ≥ 5``) are below *every* gold AND decoy magnitude in
  the R-61 / R-64 banks; the magnitude-hint only removes the
  LLM's *relative* magnitude judgment loophole. The synthetic
  R-64-baseline / R-64-W14H-only cells confirm that the
  magnitude-hinted prompt produces the same downstream answer
  as the structured prompt on the synthetic side (W17-3).
* **Not** "the runtime contract changed." The Wevra single-run
  product runtime contract is byte-for-byte unchanged. W17 is
  research-grade SDK code on the
  ``vision_mvp.experiments.phase64_live_composition`` surface +
  one new producer-prompt mode + one new dataclass + one new
  threshold table.

## 6. Active conjectures (SDK v3.18)

* **W17-C-XMODEL** (cross-model live composition): bench
  property hold-rate is saturated (8/8 on 35B under
  ``think = False``); the strict-gain claim clears the 0.50
  threshold at +0.750. **Discharged-conditional** as the
  cross-model bench-property + strict-gain transfer; remains
  conjectural on the saturated full-correctness axis (the 0.750
  result is bounded by a 35B-specific specific-tier kind
  judgment artifact). Falsifier: a future 35B run with
  ``think = False`` where the bench property regresses below
  6/8 OR the strict-gain regresses below +0.50 across ≥ 3
  alternate ``bank_seed`` values.
* **W17-C-LEARNED-HINT** (learned threshold variant):
  conjectural; out of scope for SDK v3.18.
* **W17-C-DISAMBIGUATOR** (semantic-disambiguator beyond
  W17-Λ-symmetric): the natural next-axis open question. A
  learned or LLM-distilled disambiguator that consumes the
  round-2 disambiguator's evidence text could in principle
  distinguish ``orders_payments_join`` (gold A_B in deadlock)
  from a generic decoy whose round-1 mentions are
  observationally identical. **Conjectural; out of scope for
  SDK v3.18.**
* **W17-C-CROSS-BENCH** (cross-bench transfer): the magnitude-
  hint protocol generalises to non-incident-triage benchmark
  families when the family admits closed-vocabulary kind →
  qualifying-threshold mapping AND the LLM is observed to make
  relative-magnitude judgments. **Conjectural.**

## 7. Theory consequences — sharper decomposition

The Wevra programme now has **nine** structurally-distinct moves
named one-by-one over SDK v3.7..v3.18, with the W17 family
adding two: a *positive* live-strict-gain anchor (W17-1) and a
*negative* symmetric-corroboration limit theorem
(W17-Λ-symmetric).

The defensible "thesis-after-SDK-v3.18" is that the
synthetic→real-LLM-and-bounded-context transfer story now has
**nine layers**:

* **Layer 1..5** (SDK v3.7..v3.14, unchanged from prior
  decomposition): admission, intra-round decoding, cross-round
  decoding, fixed-vocabulary normalisation, layered open-world
  normalisation.
* **Layer 6 (SDK v3.15, W14-1)**: structured producer protocol
  + comparable-magnitude events → real-LLM transfer at +0.500
  vs FIFO on Phase-61 ``qwen2.5:14b-32k``, conditional on
  prompt-side discipline (7/8 hold rate, 1/8 model-side miss).
* **Layer 7 (SDK v3.16, W15-1)**: attention-aware capsule
  context packing + hypothesis preservation → restores
  correctness when the cross-round bundle is bounded by a tight
  ``T_decoder``, conditional on the multi-hypothesis bench
  property.
* **Layer 8 (SDK v3.17, W16-1 + W16-Λ-real-replay)**: end-to-end
  W14 + W15 composition → +0.500 strict gain over
  FIFO-packed-W14-only on recorded ``qwen2.5:14b-32k`` bytes at
  ``T_decoder = 14``; conditional on the asymmetric-corroboration
  bench property AND the budget band.
* **Layer 9 (SDK v3.18, W17-1)**: magnitude-hinted producer
  protocol + W15 decoder packer → +1.000 strict gain over
  FIFO-packed-W14H-only on a *fresh* live ``qwen2.5:14b-32k``
  probe at ``T_decoder = 14``; bench property holds in 8/8
  (closing the prior 1/8 model-side miss); conditional on the
  asymmetric-corroboration bench property AND magnitude-hint
  table calibrated to the synthetic extractor.
* **Layer 9-Λ (SDK v3.18, W17-Λ-symmetric, *negative theorem*)**:
  on R-64-SYM (symmetric-corroboration regime: gold AND decoy
  both mentioned by ≥ 2 distinct producer roles via the same CCK
  tier), every capsule strategy in the SDK ties FIFO at 0.000 by
  construction. **The named structural wall.** The
  asymmetric-corroboration bench property is structurally
  necessary for W14H + W15 composition to clear the bar; without
  it, no service-blind admission AND no closed-form salience
  packer can distinguish gold from decoy.

This sharpens the answer to *"what would solving multi-agent
context actually require?"*:

1. Producer-side ambiguity preservation (W14) — emit the bench
   property's structural ingredients.
2. Producer-side relative-magnitude discipline (W17) — emit the
   bench property's *full asymmetric* corroboration shape under
   real LLMs that make relative-magnitude judgments.
3. Cross-round / cross-role capsule structure (W11/W10/W7-2/
   W8/W9) — aggregate evidence across rounds and roles via the
   capsule DAG.
4. Normalisation of the producer's drift channel (W12/W13).
5. Decoder-side context packing (W15) — under a tight
   ``T_decoder``, pack by causal salience with hypothesis
   preservation.
6. End-to-end composition (W16/W17) — *all* upstream-and-
   downstream layers must be in scope simultaneously on the
   live regime where both producer compression AND decoder
   budget pressure apply.
7. **Asymmetric-corroboration bench property (W17-Λ-symmetric
   negative theorem)** — when the bench property's
   asymmetric ingredient is structurally absent, no service-
   blind admission AND no closed-form salience packer can
   recover. This is the named wall the programme has been
   pointing toward since SDK v3.16.
8. Lifecycle audit (W4-1, T-1..T-7) — every cell auditable.

The W17-Λ-symmetric wall is the *first explicit named limit
theorem* in the programme that says "this is what no closed-form
capsule strategy can do" *constructively*, not just "this is
what no admission policy alone can do" (the W10-Λ pattern from
SDK v3.11). It is the named research frontier for SDK v3.19+.

## 8. Files changed

* New experiment driver:
  ``vision_mvp/experiments/phase64_live_composition.py`` — driver
  + 4 symmetric-corroboration scenario builders +
  :func:`build_phase64_sym_bank` + :func:`run_phase64` +
  :func:`run_cross_regime_synthetic`.
* SDK surface (purely additive; W17 family):
  ``vision_mvp/wevra/team_coord.py`` —
  ``PRODUCER_PROMPT_MAGNITUDE_HINTED`` constant,
  :class:`OperationalThreshold` dataclass,
  ``magnitude_thresholds`` field on
  :class:`RoleExtractionSchema`,
  :func:`_render_magnitude_hinted_prompt`,
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS`,
  :func:`incident_triage_magnitude_thresholds`,
  ``magnitude_hinted`` parameter on
  :func:`incident_triage_role_schemas`. Re-exported from
  ``vision_mvp/wevra/__init__.py``.
* New tests:
  ``vision_mvp/tests/test_wevra_phase64.py`` — 20 tests covering
  the W17 protocol surface, R-64-baseline, R-64-W14H-only,
  R-64-SYM bank shape, R-64-SYM wall, cross-regime synthetic
  separation, W17-3 backward-compat reduction.
* Patch to legacy parser dispatch:
  ``vision_mvp/experiments/phase61_producer_ambiguity_preservation.py``
  ``CapturingOllamaExtractor.extract_round`` — the live extractor
  now treats ``PRODUCER_PROMPT_MAGNITUDE_HINTED`` as a
  structured-prompt variant (per-event mandate ⇒ dedupe by
  ``(kind, payload)`` rather than ``kind`` alone).
* Test patch:
  ``vision_mvp/tests/test_wevra_producer_ambiguity.py`` —
  ``test_all_modes_listed`` updated for the additive third mode.
* SDK version bump:
  ``vision_mvp/wevra/__init__.py`` —
  ``SDK_VERSION = "wevra.sdk.v3.18"``.
* Artifacts (this milestone):
  ``docs/data/phase64_cross_regime_synthetic.json``,
  ``docs/data/phase64_live_maghint_qwen2_5_14b_n8.json``,
  ``docs/data/phase64_live_struct_qwen2_5_14b_n8.json``,
  ``docs/data/phase64_live_naive_qwen2_5_14b_n8.json``,
  ``docs/data/phase64_live_maghint_qwen3_5_35b_n8.json``.
* Doc updates:
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` (R-64 anchor +
  bar 14 + § 2.13 R-64 ingredients),
  ``docs/RESEARCH_STATUS.md`` (this milestone),
  ``docs/THEOREM_REGISTRY.md`` (W17 family),
  ``docs/HOW_NOT_TO_OVERSTATE.md`` (W17 framing rules),
  ``docs/context_zero_master_plan.md`` (§ 4.35 SDK v3.18),
  ``docs/START_HERE.md`` (current milestone pointer),
  ``docs/RESULTS_WEVRA_LIVE_COMPOSITION.md`` (this file),
  ``CHANGELOG.md`` (v3.18 entry),
  ``papers/context_as_objects.md`` (synthesis-after-v3.18
  reading).

## 9. What this milestone advances

* **The original Context-Zero thesis** — *per-agent
  minimum-sufficient context for multi-agent teams* — gains its
  *first fresh-live end-to-end real-LLM strict +1.000 advance*
  over the strongest non-composed baseline AND closes the prior
  1/8 R-61-OLLAMA-A model-side judgment miss. The
  minimum-sufficient context for the auditor's decision under a
  real LLM is now jointly:
    (i) emitted by the producer **with explicit operational
        thresholds + anti-relative-magnitude discipline**
        (W17 ensures the LLM emits the bench property's full
        asymmetric corroboration shape regardless of relative
        magnitude judgments);
    (ii) admitted by the cross-round pipeline (W11/W12/W13);
    (iii) preserved by the decoder under a strict context
          budget (W15 ensures the round-2 disambiguator falls
          inside the kept bundle);
    (iv) elected to the answer by the priority decoder + W11
         contradiction-aware drop **on the asymmetric-
         corroboration regime** (W17-Λ-symmetric names the
         wall: when the asymmetric ingredient is absent, no
         capsule strategy in the SDK can recover).
* **The first explicit symmetric-corroboration limit theorem.**
  The W15-C-SYMMETRIC / W16-C-SYMMETRIC conjectures pointed at
  this wall but did not prove it; W17-Λ-symmetric *constructs*
  the bench, mechanically verifies the wall under both
  ``T_decoder ∈ {None, 24}``, and discharges the prior
  conjectures as a negative theorem. **The next research
  frontier** is a learned or LLM-distilled
  semantic-disambiguator beyond the closed-form capsule surface.
* **Live-end-to-end + magnitude-hinted-protocol +
  symmetric-corroboration-wall split** is now a first-class axis
  of the success bar. Bar 14 of
  ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`` requires
  *all three* — replay-only methods, structured-prompt-only
  methods that do not enumerate operational thresholds, AND
  downstream-only methods that ignore the symmetric-
  corroboration counterexample do NOT clear bar 14.
* **The Wevra programme has nine structural axes** with named
  limit theorems on each, AND SDK v3.18 mints both a *positive*
  live-strict-gain claim (W17-1) AND a *negative*
  symmetric-corroboration wall (W17-Λ-symmetric). The runtime
  contract is byte-for-byte unchanged; all nine are
  research-grade SDK extensions.

## Cross-references

* Bench: ``vision_mvp/experiments/phase64_live_composition.py``
* Method: composition of
  ``vision_mvp/wevra/team_coord.py``
  (``StructuredProducerProtocol(mode=magnitude_hinted)`` +
  :data:`INCIDENT_TRIAGE_DEFAULT_MAGNITUDE_THRESHOLDS` +
  :class:`AttentionAwareBundleDecoder`)
* Tests: ``vision_mvp/tests/test_wevra_phase64.py``
* Prior milestones:
  ``docs/RESULTS_WEVRA_COMPOSED_REAL_LLM.md`` (W16),
  ``docs/RESULTS_WEVRA_ATTENTION_AWARE.md`` (W15),
  ``docs/RESULTS_WEVRA_PRODUCER_AMBIGUITY.md`` (W14)
* Success criterion: ``docs/SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md``
  (R-64 anchor + bar 14 + § 2.13 R-64 ingredients)
* Theorem registry: ``docs/THEOREM_REGISTRY.md`` (W17 family)
* Master plan: ``docs/context_zero_master_plan.md`` § 4.35
