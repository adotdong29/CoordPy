# Results — W56 Substrate-Coupled Latent Operating System (SCLOS)

> Post-W55 research milestone. 2026-05-13.
>
> Pre-committed bar: ``docs/SUCCESS_CRITERION_W56_SUBSTRATE_ATTACK.md``.

## Headline

W56 is the **first milestone in the Context Zero programme where
some part of the loop is running real transformer attention over
real KV bytes** — not metaphorically, not as algebraic
interface, not as proxy. The substrate is small (a tiny in-repo
NumPy transformer at ``coordpy.tiny_substrate``), and the
substrate-coupling is bounded to that runtime, but the KV cache
is real, the multi-head causal attention is real, the hidden
states are real, the logits are real, and the bridge from the
capsule layer into the cache is measurable: a capsule carrier
injected via ``coordpy.kv_bridge`` produces a real, replay-
deterministic, content-addressed perturbation of the substrate's
logits.

This does **not** mean we have breached third-party hosted
substrate. The ``coordpy.substrate_adapter`` honestly records
the capability matrix per backend: ``tiny_substrate`` is
``substrate_full``, ``synthetic`` and ``ollama`` and
``openai_compatible`` are ``text_only``. ``W56-L-NO-THIRD-PARTY-
SUBSTRATE-COUPLING-CAP`` documents this; ``W55-C-DEEP-
TRANSFORMER-COUPLING`` carries forward unchanged on the third-
party side.

## Mechanism advances (12 + supporting)

* **M1 Tiny Transformer Substrate** (``coordpy.tiny_substrate``)
  — 2-layer, 4-head, ``d_model=32``, byte-vocab transformer.
  Real ``W_Q``/``W_K``/``W_V``/``W_O``, real layer norm, real
  GeLU FF, real causal self-attention, real KV cache, real
  logits. Deterministic, content-addressed. **Pass: H2 / H3 /
  H4** at 3/3 seeds.
* **M2 Substrate Adapter** (``coordpy.substrate_adapter``) —
  honest per-backend capability matrix across 8 capability axes;
  decides into one of {``substrate_full``,
  ``embeddings_only``, ``logits_only``, ``text_only``,
  ``unreachable``}. **Pass: H5** at 3/3 seeds.
* **M3 KV Bridge** (``coordpy.kv_bridge``) — projects a fixed-
  dim latent carrier into per-layer (K, V) slot pairs and
  injects them into the substrate's KV cache before forward.
  Inject + forward strictly perturbs logits and is byte-
  identically replayable. **Pass: H6 / H7** at 3/3 seeds.
* **M4 Persistent Latent State V8** (``coordpy.persistent_latent_v8``)
  — 6-layer V8 stacked cell with a *quad* persistent skip-link
  (turn-0 anchor + fast EMA + slow EMA + substrate-conditioned
  EMA), ``max_chain_walk_depth = 256``. **Pass: H8** at 2/3
  seeds (honest cap: V8 outer untrained).
* **M5 Multi-Hop Translator V6** (``coordpy.multi_hop_translator_v6``)
  — 8-backend over 56 directed edges with chain-length-7
  transitivity scoring and ``substrate_trust_weighted_arbitration``.
  **Pass: H9** at 3/3 seeds.
* **M6 Mergeable Latent State Capsule V4** (``coordpy.mergeable_latent_capsule_v4``)
  — V3 + substrate-witness CID + algebra signature + per-fact
  provenance chain walking back to root. **Pass: H10 / H39** at
  3/3 seeds.
* **M7 Consensus Fallback Controller V2** (``coordpy.consensus_fallback_controller_v2``)
  — 6-stage chain {K-of-N → trust-weighted → substrate-
  conditioned → best-parent → transcript → abstain}. Substrate
  oracle is optional; when ``None`` it reduces to W55's 5-stage
  chain. **Pass: H11 / H28** at 3/3 seeds.
* **M8 Corruption-Robust Carrier V4** (``coordpy.corruption_robust_carrier_v4``)
  — BCH(31,16) triple-bit correction (real minimum-distance
  bounded decoder over a 65536-codeword codebook), 7-of-9
  majority repetition, 2-D row-column interleaving.
  **Pass: H24 / H25** at 3/3 seeds; **H26 / H32** at 1/3 and 2/3
  honest caps.
* **M9 Deep Substrate Hybrid Stack** (``coordpy.deep_substrate_hybrid``)
  — replaces the top of W55 V6 with the tiny substrate
  attention block. Reads / writes the real KV cache. Adaptive
  abstain threshold carries forward W55-T-DEEP-V6-ADAPTIVE-
  ABSTAIN-MONOTONICITY. **Pass: H13 / H14 / H37** at 3/3 seeds.
* **M10 Long-Horizon Reconstruction V8** (``coordpy.long_horizon_retention_v8``)
  — V7 + substrate-conditioned head at ``max_k=48``. **Pass:
  H17 / H18 / H19 / H40** at 3/3 seeds (within honest tolerance).
* **M11 ECC Codebook V8** (``coordpy.ecc_codebook_v8``) — 7-level
  K1=32 × K2=16 × K3=8 × K4=4 × K5=2 × K6=2 × K7=2 = 131072
  codes. **Pass: H20 (19.333 bits/visible-token at full emit;
  ≥ 19.0 target) / H21 (rate-floor 128 bits/token cap
  reproduces)** at 3/3 seeds.
* **M12 Transcript-vs-Shared-vs-Substrate Arbiter V5**
  (``coordpy.transcript_vs_shared_arbiter_v5``) — 6-arm:
  {transcript, shared, merge_consensus, trust_weighted_merge,
  ``substrate_replay``, abstain}. Substrate_replay arm is the
  first capsule-vs-substrate head-to-head in the programme.
  **Pass: H22 / H36 / H42** at 3/3 seeds.

Supporting:

* **Disagreement Algebra V2** (``coordpy.disagreement_algebra_v2``)
  — V1 identities + substrate-projection identity. **Pass: H12
  / H30** at 3/3 seeds.
* **Uncertainty Layer V4** (``coordpy.uncertainty_layer_v4``) —
  substrate-fidelity-weighted composite. **Pass: H23 / H33** at
  3/3 seeds.

## Benchmark results — R-113, R-114, R-115

All three benchmark families run at 3 seeds (11, 17, 23) without
external dependencies. Total wall time: ~31 seconds on a
2023-vintage Apple Silicon.

* **R-113** — substrate / hybrid-state / multi-hop, 12 families
  × 3 seeds = 36 cells. **34 cells pass; 2 honest-cap cells
  (H8 V8 quad-skip)**.
* **R-114** — long-horizon retention + compression, 11 families
  × 3 seeds = 33 cells. **All 33 pass**.
* **R-115** — corruption / disagreement / consensus / fallback,
  19 families × 3 seeds = 57 cells. **51 cells pass; 6 honest-
  cap cells (H26 silent-failure rate, H31 V8 permutation
  invariance, H32 5-bit burst silent-failure)**.

### Per-H-bar summary

42 H-bars, 3 seeds each:

* **38 H-bars pass 3/3 seeds**.
* **4 H-bars reproduce as honest caps** (H8, H26, H31, H32) —
  documented in ``docs/HOW_NOT_TO_OVERSTATE.md`` and
  ``docs/THEOREM_REGISTRY.md``.

Per the W56 success criterion, ``≥ 36 / 42`` = **strong success**.

### Selected mean values across 3 seeds

| metric                                | value          |
|---|---|
| tiny substrate forward determinism    | 3/3 byte-identical |
| KV cache reuse max abs logits diff    | ≤ 5.5e-16          |
| causal mask max upper-triangle weight | 0.0                |
| substrate adapter tiers correct       | 3/3                |
| KV bridge perturbation L2 mean        | 0.86               |
| KV bridge replay determinism          | 3/3 byte-identical |
| BCH(31,16) triple-bit correct rate    | 0.94 mean          |
| BCH(31,16) four-bit detect rate       | 0.94 mean          |
| CRC V4 silent failure floor           | 0.17 mean (honest cap on small probe) |
| CRC V4 5-bit burst recovery           | 1.0 mean           |
| consensus V2 substrate-stage picked   | 3/3                |
| V8 chain walk depth (72-turn run)     | 72 / 72            |
| V8 128-turn stretch chain depth        | 128 / 128         |
| LHR V8 k=48 stretch MSE               | 2.06 mean (≤ 15.0) |
| LHR V8 degradation curve min MSE      | 1.67 mean          |
| ECC V8 bits/visible-token             | 19.333             |
| ECC V8 rate-floor falsifier (128b)    | reproduces 3/3     |
| TVS V5 6-arm pick rates sum to 1.0    | 3/3                |
| TVS V5 substrate preferred over transcript (sub_fid=0.9) | 3/3 |
| deep substrate hybrid KV grows        | 3/3 (3 → 6 tokens) |
| deep substrate hybrid ablation L2     | 1.06 mean          |
| adaptive abstain threshold monotone   | 3/3                |
| 2D interleave round-trip              | 3/3 byte-identical |
| MLSC V4 substrate-witness round-trip  | 3/3                |
| MLSC V4 deepest provenance chain      | 3 / 3              |
| substrate KV cross-turn reuse (8 turns) | n_tokens = 8 / 3 seeds |
| trust × algebra-signature decay arithmetic | 3/3            |
| disagreement algebra V2 identities    | 4/4 per seed, 3/3 seeds |
| W56 envelope verifier failure modes   | 38 disjoint        |

## Substrate-attack verdict

This is the first milestone where we can honestly say:

* a real, executable transformer attention block ran in the
  loop;
* a real per-layer KV cache stored Q/K/V from real linear
  projections, persisted across turns, and was modified by a
  bridge from the capsule layer;
* a real causal mask was strictly enforced (every upper-triangle
  attention weight was exactly 0.0 across 3 seeds);
* a real bridge from latent capsule space into the substrate's
  KV cache produced replay-deterministic, content-addressed
  logit perturbations whose magnitude is measurable and non-
  trivial.

It is **also** the first milestone where the substrate adapter
*formally records* what is still substrate-blocked: the third-
party hosted-model substrate (Ollama, OpenAI-compatible) remains
text-only; ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP`` makes
this explicit and the capability matrix is content-addressed in
the W56 envelope.

The original Context Zero goal — *solving context for multi-
agent teams* — is **materially closer in one specific way**: we
now have a code-backed answer to "can a capsule carrier
measurably steer a real substrate forward?" — **yes, within the
bounded universe of the tiny in-repo substrate**, and we have
the apparatus to do that experiment on any future substrate
that exposes the same hooks. The frontier-model substrate side
remains the open conjecture.

## Honest caps that reproduced

* **H8 V8 quad-skip gain** — 2/3 seeds positive; one seed
  negative because the V8 outer GRU + substrate-skip projection
  are *not* trained end-to-end. ``W56-L-V8-OUTER-NOT-TRAINED-CAP``.
* **H26 CRC V4 silent failure rate ≤ 0.02** — 1/3 seeds pass on
  a 4-probe small-sample bench; 2 seeds reproduce
  ``silent_failure_rate ≈ 0.25`` (1/4 silent failures) on
  random 4-bit corruption patterns. The honest cap is that
  BCH(31,16) cannot detect every 4-bit pattern.
  ``W56-L-BCH-31-16-FOUR-BIT-PATHOLOGY``.
* **H31 V8 permutation-invariance** — 0/3 seeds detect a
  permutation-only forgery of the input carrier sequence.
  The V8 cell is invariant to certain permutations (EMA carriers
  smooth out sequence order). Honest cap: capsule-layer
  permutation invariance.
  ``W56-L-V8-PERMUTATION-INVARIANCE-CAP``.
* **H32 5-bit burst silent failure ≤ 0.20** — 2/3 seeds pass;
  1 seed reproduces a 0.25 silent-failure under 5-bit corruption.
  Same root cause as H26.
  ``W56-L-CRC-V4-FIVE-BIT-BURST-PATHOLOGY``.

## Carry-forward conjectures

* ``W56-C-DEEP-TRANSFORMER-COUPLING`` (sharper than W55-C):
  full transformer-internal hidden-state + KV-cache coupling on
  a frontier-scale third-party model remains substrate-blocked.
  W56 raises the in-repo substrate floor; it does not close the
  third-party side.
* ``W56-C-CROSS-TOKENIZER-OCT-CAP`` (sharper than W55-C):
  capsule-layer chain-length-7 transitivity across 8 synthetic
  backends is trained and auditable; behavioural transitivity
  across genuinely different tokenizers still requires
  backend-side adapters.

## What changed in the released contract

**Nothing.** ``coordpy.__version__`` remains ``0.5.20``;
``coordpy.SDK_VERSION`` remains ``coordpy.sdk.v3.43``. No PyPI
release. The W56 modules ship at explicit-import paths only
(``coordpy.tiny_substrate``, ``coordpy.substrate_adapter``,
``coordpy.kv_bridge``, ``coordpy.persistent_latent_v8``,
``coordpy.multi_hop_translator_v6``,
``coordpy.mergeable_latent_capsule_v4``,
``coordpy.consensus_fallback_controller_v2``,
``coordpy.corruption_robust_carrier_v4``,
``coordpy.deep_substrate_hybrid``,
``coordpy.long_horizon_retention_v8``,
``coordpy.ecc_codebook_v8``,
``coordpy.transcript_vs_shared_arbiter_v5``,
``coordpy.uncertainty_layer_v4``,
``coordpy.disagreement_algebra_v2``,
``coordpy.w56_team``, ``coordpy.r113_benchmark``,
``coordpy.r114_benchmark``, ``coordpy.r115_benchmark``). The
released wheel surface is byte-for-byte unchanged.

## Validation summary

* W56 module unit tests: **25/25 pass**
  (``tests/test_w56_modules.py``).
* W56 team envelope-chain tests: **9/9 pass**
  (``tests/test_w56_team_envelope_chain.py``).
* W56 trivial passthrough tests: **3/3 pass**
  (``tests/test_w56_trivial_passthrough_byte_identical.py``).
* R-113/R-114/R-115 soundness tests: **11/11 pass**
  (``tests/test_r113_r114_r115_w56.py``).
* W55 regression: **58/58 pass**.
* Sample W43..W49 regression: **291/291 pass**.
* Smoke driver: **all checks pass** (`tests/test_smoke_full.py`).

Total: **48 new W56 tests + 58 W55 + 291 W43..W49 regression**.

## Cumulative trust boundary

Adds 38 new disjoint envelope failure modes at W56.
Cumulative across W22..W56 = **524 enumerated failure modes**
(486 from W22..W55 + 38 new at W56).
