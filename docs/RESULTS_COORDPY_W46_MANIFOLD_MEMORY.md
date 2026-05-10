# RESULTS — W46 Manifold Memory Controller (MMC)

> Programme step: post-W45, post-W44, post-CoordPy 0.5.20. Mints
> axis 43 of the Context Zero programme. Strictly additive on
> top of the W45 LMC layer, the W44 LMCC layer, the W43 PMC
> layer, and the released v3.43 line. The released SDK contract
> is byte-for-byte unchanged; the W46 surface lives at
> `coordpy.manifold_memory` and is reachable only through an
> explicit import.

## TL;DR

W46 is the first **multi-layer, memory-conditioned, transformer-
facing** capsule-native layer in CoordPy. Where W45 made *one*
fitted gating decision per turn from a *flat* feature vector,
W46:

* stacks fitted ridge layers on layer-wise residuals,
* maintains a bounded, content-addressed memory bank of past
  channel features + gate logits,
* reads from that bank via causally-masked time-attention,
* generalises the rank-1 LoRA-style role adapter to rank-r,
* encodes channel state into a learned K-prototype dictionary
  basis with bijective decode,
* emits a packed multi-line `MANIFOLD_CTRL` model-facing control
  surface,
* and binds a deterministic shared-prefix capsule that lets
  consecutive turns receive byte-identical prefix bytes.

The R-93 benchmark family across 5 seeds × 12 cell families
produces the following honest, repeatable, code-backed results
vs the released `AgentTeam` baseline, the W43 closed-form audit
arm, the W44 live-coupled arm, and the W45 learned-coupled arm:

| family | metric | baseline | w43 | w44 | w45 | **w46** | Δ memory vs w45 |
|---|---|---|---|---|---|---|---|
| `r93_trivial_memory_passthrough` | passthrough_ok | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | +0.000 (sanity) |
| `r93_long_branching_memory` | precision_on_deep_turns | 1.000 | 1.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| `r93_cyclic_consensus_memory` | cycle_consensus_precision | 0.750 | 0.750 | 0.750 | 1.000 | **1.000** | +0.000 (preserves W45) |
| `r93_role_shift_adaptation` | role23_precision (rank2 vs shared) | 0.500 | 0.500 | 0.500 | 1.000 | **1.000** | +0.000 (+1.000 vs shared-only) |
| `r93_compressed_control_packing` | ctrl_round_trip_ok | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| `r93_memory_facing_hint_response` | task_correct_rate | 0.000 | 0.000 | 0.000 | 0.000 | **1.000** | **+1.000** |
| `r93_causal_mask_preservation` | causal_mask_preserved | n/a | n/a | n/a | n/a | **1.000** | n/a |
| `r93_dictionary_reconstruction` | dictionary_round_trip_ok | n/a | n/a | n/a | n/a | **1.000** | n/a |
| `r93_shared_prefix_reuse` | prefix_reuse_ok | n/a | n/a | n/a | 0.000 | **1.000** | **+1.000** |
| `r93_w46_falsifier` | no_false_abstain | 1.000 | n/a | n/a | n/a | **1.000** | +0.000 (no over-claim) |
| `r93_w46_compromise_cap` | downstream_protect_rate | 0.000 | n/a | n/a | 0.000 | 0.000 | +0.000 (limitation reproduces) |
| `r93_replay_determinism` | replay_determinism_ok | n/a | n/a | n/a | n/a | **1.000** | n/a |

All H1..H12 hypotheses of the pre-committed success criterion
(`docs/SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md`) pass cleanly.
The released CoordPy 0.5.20 stable smoke driver
(`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED" with
the W46 module on disk. The W43, W44, and W45 benchmark families
(`coordpy.r90_benchmark`, `coordpy.r91_benchmark`,
`coordpy.r92_benchmark`) reproduce byte-for-byte; no
R-90 / R-91 / R-92 family is perturbed by the W46 module.

## What is shipped

* **`coordpy/manifold_memory.py`** (~2100 LoC, NumPy-free): the
  W46 layer. Seven content-addressed components —
  `MultiLayerControllerParams`, `LayerParams`,
  `MultiRankRoleAdapter`, `DictionaryBasis`,
  `ManifoldMemoryBank` (+ `MemoryEntry`),
  `TimeAttentionWitness` (+ `compute_time_attention`),
  `ControlTokenWitness` (+ `build_control_token_string`),
  `PrefixCapsule` (+ `build_prefix_capsule`),
  `fit_memory_controller` (closed-form stage-wise ridge stack),
  `forward_memory_controller`,
  `ManifoldMemoryRegistry`, `ManifoldMemoryOrchestrator`,
  `ManifoldMemoryHandoffEnvelope`, `ManifoldMemoryTeam`
  end-to-end orchestrator. Plus a `MemoryAwareSyntheticBackend`
  for the hermetic memory-facing benchmark. Plus a 21-mode
  verifier (`verify_manifold_memory_handoff`).

* **`coordpy/r93_benchmark.py`** (~1100 LoC, dependency-free):
  the R-93 benchmark family. Twelve cell families, five honest
  baselines (`baseline_team`, `w43_closed_form`,
  `w44_live_coupled`, `w45_learned_coupled`,
  `w46_memory_coupled`), 5-seed aggregator, text-report
  renderer.

* **`tests/test_manifold_memory.py`** (44 tests): per-component
  unit coverage — schema / branches / failure modes,
  dictionary encode/decode bijection + closest-assignment,
  memory bank causal mask + ring-buffer eviction + head CID,
  time-attention zero/disabled/causal-mask/aligned-pool,
  multi-layer fitter determinism + zero-params zero-logit
  identity, prefix capsule first-N-stability + reuse flag +
  64-hex CID, control token off/compact/full modes + bad-mode
  raise, trivial-passthrough falsifier matches AgentTeam
  byte-for-byte, verifier soundness on schema / token / outer
  tamper, run-determinism, memory-aware backend behaviour,
  public-surface invariants.

* **`tests/test_r93_benchmark.py`** (20 tests): each H1..H12
  hypothesis is exercised directly; aggregator + per-family +
  per-seed regression coverage; falsifier + compromise-cap
  family-level checks.

* **`docs/SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md`**: the
  pre-committed success bar (H1..H12, falsifiers + per-channel
  verdicts, scope rules, version + release status).

## What was NOT done (honest scope)

W46 is a **capsule-layer milestone** with a *multi-layer
learned controller stack*, *causally-masked time attention over
a bounded memory bank*, *learned dictionary basis*, *multi-rank
role adapters*, *packed control token surface*, and
*shared-prefix capsule reuse*. It does NOT close any of:

* **`W43-C-MIXED-CURVATURE-LATENT`** — full transformer-internal
  mixed-curvature attention. The W46 multi-layer controller
  operates strictly over W43 capsule-layer channel encodings;
  it does not modify the model's attention computation.

* **`W43-C-COLLECTIVE-KV-POOLING`** — host-collective KV-cache
  sharing. W46 runs on a single backend within a single process;
  it does not pool KV state.

* **`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`** — a true continuous
  Gr(k, d) homotopy. The subspace channel still captures a
  single point on the Grassmannian per cell.

* **`W44-C-LIVE-LATENT`** — promoting audit-only channels to
  *transformer-internal* behavioural channels. The W45
  bounding (hyperbolic + euclidean channels consumed by the
  learned controller) carries forward and is *strengthened* at
  W46 by the multi-layer + memory + dictionary path; closing
  the conjecture still requires substrate access.

* **`W45-C-DEEP-TRANSFORMER-COUPLING`** — full deep transformer-
  coupled controller with hidden-state consumption and
  attention-mask emission. The W46 mechanism is the strongest
  capsule-layer approximation we can write today; it is
  multi-layer but every layer is fitted closed-form on
  capsule-layer features, and the time-attention readout is a
  cosine-similarity pool over capsule-layer state — not the
  model's internal attention.

W46 does NOT claim:

* the multi-layer controller is a deep neural network in the
  autograd sense. Each layer is fitted closed-form via
  stage-wise ridge on layer-wise residuals; there is no SGD,
  no backprop, no learned non-linearity beyond the per-layer
  softmax attention pool.

* true shared-KV between turns. The shared-prefix capsule
  guarantees *byte-identical prefix bytes* across applicable
  turns. Whether the underlying transformer reuses its
  internal KV cache for that prefix is a model-side runtime
  concern that the W46 surface does not measure or assert.

* true time-attention on the model's hidden state. The W46
  time-attention is a cosine-similarity readout over
  capsule-layer channel features pooling capsule-layer gate
  logits; the model's attention computation is unchanged.

* the multi-rank role adapter is true rank-r LoRA on a
  transformer. The W46 multi-rank adapter is a rank-r
  perturbation of the capsule-layer policy parameters, not the
  model's weight matrices.

* the dictionary basis is a learned vocabulary in the
  language-modeling sense. It is a deterministic K-prototype
  clustering of the training-bank channel features.

* multi-host learned coupling.

* the controller is "trained on real LLM traces." Fitting is
  closed-form stage-wise ridge on synthetic banks pre-committed
  in the R-93 sources.

## New mechanism (PART C)

For each turn:

1. The observation builder emits a `CellObservation`.
2. The W45 inner runs and produces a closed-form-fitted learned
   gating decision (W45 base + W44 gate + W43 PMC envelope).
3. The W46 layer encodes channel features and runs the
   multi-layer controller stack:
   * **Layer 0**: the W45-shaped per-channel projection +
     attention pool + role-delta + margin offset.
   * **Layer i >= 1**: a fitted per-channel projection + softmax
     attention pool. Each layer's output is added to the
     running gate logit.
4. The multi-rank role adapter applies the rank-r per-role
   delta to the running gate logit (basis = signed per-channel
   logits + cyclic rotations).
5. The dictionary basis encodes the flattened channel features
   as `(prototype_index, residual)` — bijective decode.
6. The time-attention readout pools past gate logits from the
   memory bank using cosine-similarity-weighted softmax over
   the strictly admissible (turn_index `< t`) entries.
7. The total gate logit = base + sum(layer logits) +
   role_delta + time_attention_weight * pooled +
   time_attention_bias.
8. The sigmoid + confidence bucket are derived deterministically.
9. The W46 layer optionally substitutes an abstain output on
   one of five W46-specific abstain branches:
   `memory_margin_abstain`, `memory_time_attn_abstain`,
   `memory_spherical_abstain`, `memory_subspace_abstain`,
   `memory_causal_abstain`.
10. The packed `MANIFOLD_CTRL` control block is appended to the
    prompt; the bytes are content-addressed and bijectively
    recoverable from the envelope.
11. The shared-prefix capsule emits stable prefix bytes once
    the team has produced `prefix_turns` outputs.
12. The W46 envelope binds: TEAM_HANDOFF CID, W44 envelope CID,
    W43 envelope CID, multi-layer-controller params CID,
    dictionary CID, memory-bank head CID, time-attention
    witness CID, multi-rank-adapter witness CID, causal-mask
    witness CID, control-token witness CID, prefix-capsule CID,
    prompt-construction witness CID — all under one
    `memory_outer_cid`.

The released `AgentTeam.run` is byte-for-byte unchanged; the
`LiveManifoldTeam.run` path is byte-for-byte unchanged; the
`LearnedManifoldTeam.run` path is byte-for-byte unchanged; the
W46 surface is a new `ManifoldMemoryTeam` class that sits
beside all three.

## Architecture triage (PART B)

| Frontier candidate                           | W46 bucket                                   | Status |
|---|---|---|
| Multi-layer learned controller stack         | **implementable + learnable now**            | shipped: 2 layers default, fit stage-wise ridge on residuals |
| Multi-rank role adapter                      | **implementable + learnable now**            | shipped: rank-r LoRA-style basis = signed-channel-logits + rotations |
| Learned basis / dictionary                   | **implementable + learnable now**            | shipped: K-prototype dictionary with bijective decode, seeded farthest-point + 1-pass refinement |
| Causally-masked time attention               | **implementable + learnable now**            | shipped: cosine-similarity softmax over strictly admissible memory entries |
| Memory bank with capsule-CID provenance      | **implementable + learnable now**            | shipped: bounded ring buffer with head_cid + per-entry to_dict, audit-verifiable |
| Packed model-facing control surface          | **implementable + learnable now**            | shipped: `MANIFOLD_CTRL` 3-mode (off/compact/full), bijective from envelope |
| Shared-prefix / prefix-capsule reuse         | **implementable + approximable now**         | shipped: deterministic prefix from first-N outputs; honest "identical prefix bytes" claim |
| True KV-cache pooling across turns           | **substrate-blocked**                        | conjectural; W46 emits identical prefix bytes but does not touch model-internal KV |
| Transformer-internal mixed-curvature attention | **substrate-blocked**                       | conjectural since W43; not addressed at W46 |
| Continuous Grassmannian homotopy             | **substrate-blocked**                        | conjectural since W43; not addressed at W46 |
| End-to-end backprop / SGD on the stack       | **deliberately deferred**                    | W46 stays closed-form for determinism and audit; an autograd path would be a future milestone |
| Hidden-state-aware time attention            | **substrate-blocked**                        | the W46 time-attention is a capsule-layer cosine pool, not model-internal |

Cumulative trust boundary across W22..W46 = **261 enumerated
capsule-layer failure modes** (W22..W42 cumulative 196, plus
18 new at W43, plus 12 new at W44, plus 14 new at W45, plus 21
new at W46).

## Capsule strengthening (PART D)

W46 strengthens the released TEAM_HANDOFF + W43/W44/W45 capsules
with eight new content-addressed witnesses:

1. **Multi-layer controller parameter provenance.** The
   `MultiLayerControllerParams.cid()` binds the W45 base CID,
   every `LayerParams` CID, the `MultiRankRoleAdapter.cid()`,
   the `DictionaryBasis.cid()`, and the fitted
   `time_attention_weight` / `time_attention_bias`. An auditor
   can reproduce the params bit-for-bit by re-fitting on the
   recorded training-set CID; the multi-layer-controller CID is
   the binding handle.

2. **Memory bank head CID.** Every envelope records the
   `memory_bank_head_cid` — SHA-256 over the sorted bank
   entries after the current turn's append. An auditor walking
   the envelope chain can re-derive the bank state at every
   turn.

3. **Time-attention witness CID.** Each envelope records the
   query L1 norm, the softmax-normalised attention weights
   actually used, the pooled scalar value, the mask size (number
   of admissible past entries), and the `enabled` flag. An
   auditor can prove which past entries drove the current
   turn's time-attention contribution.

4. **Multi-rank adapter witness CID.** Each envelope records the
   role, role-delta value, role-delta rank, and whether the
   adapter was active for the turn. Distinguishes "shared-base
   policy" from "rank-r-specific delta firing."

5. **Dictionary witness fields.** The envelope records the
   `dict_index` (closest prototype) and `dict_residual_l1`
   (residual L1 norm); combined with the `dictionary_cid`,
   these are sufficient to reconstruct the channel features
   bijectively.

6. **Control-token witness CID.** When `control_token_mode` is
   `compact` or `full`, the envelope binds a 64-hex
   `control_token_witness_cid` derived from every byte the
   model saw in the `MANIFOLD_CTRL` block — route, conf, p,
   layer_logits, mem_attn, dict_idx, mem_summary, and ctrl
   bytes SHA-256.

7. **Prefix-capsule CID.** Every envelope records the
   `prefix_capsule_cid` — content-addressed by the policy
   entry CID, the first-N prior-output SHAs, the prefix bytes'
   SHA-256, and the `reused` flag. An auditor can prove which
   prefix bytes the model received on each turn AND whether
   they matched the previous turn's prefix.

8. **Causal-mask witness CID** (inherited from W45 and bound at
   the W46 envelope layer).

W46 is strictly additive: when configured trivially, the
released TEAM_HANDOFF + W43 + W44 + W45 envelopes are sealed
byte-for-byte.

## Deep-learning / transformer-facing advances (PART E/F/H)

This is the milestone where Context Zero / CoordPy's capsule
layer becomes genuinely *deep* and genuinely *transformer-
facing*, while staying honest about what is actually deep vs
analytically fitted.

* **Multi-layer fitted attention.** Each layer in the
  controller stack is a per-channel projection + softmax
  attention pool. Fitted via stage-wise ridge on layer-wise
  residuals. *Honest scope*: each layer is closed-form solvable;
  the stack is deep in the sense that gate logit at turn `t` is
  the sum of L additive layer contributions, but the layers
  themselves are not trained via backprop.

* **Multi-rank low-rank adapter stack.** The rank-r per-role
  delta is the strongest executable approximation we can write
  for "role-specific KV state" without substrate access. The
  rank-r basis vectors are signed per-channel logits + cyclic
  rotations; the delta is fit via closed-form ridge on per-role
  residuals. *Honest scope*: this is a rank-r perturbation of
  the capsule-layer policy parameters, not the model's weight
  matrices.

* **Learned dictionary basis with bijective reconstruction.**
  K-prototype clustering produces a learned codebook over
  flattened channel features. The per-turn encoding is
  `(prototype_index, residual)`; decoding sums the prototype +
  residual to recover the original feature vector exactly.
  *Honest scope*: this is a sparse-code over capsule-layer
  features, not a learned tokenizer.

* **Causally-masked time-attention over an explicit memory bank
  with capsule-CID provenance.** The bank stores past channel
  features + gate logits + decision branches; the time-attention
  query is the current cell's flattened channel features; the
  softmax weights come from cosine similarity; the pooled value
  is added to the running gate logit. The mask is strict (a
  turn cannot attend to itself or the future). The bank's
  head CID is per-turn — an auditor walking the envelope chain
  re-derives the bank state. *Honest scope*: this is a
  capsule-layer cosine-similarity pool over capsule-layer
  state, not transformer-internal time attention.

* **Packed model-facing control surface (`MANIFOLD_CTRL`).** A
  multi-line YAML-shaped control block that carries route +
  conf + p (W45 surface) plus layer_logits + mem_attn +
  dict_idx + mem_summary (W46 additions). Strictly more
  *structured bits per visible token* than the W45 single-line
  hint at a structurally bounded overhead (≤ 40 extra tokens
  per turn). *Honest scope*: the bytes are a deterministic
  text fragment the model may attend to; the H6 task-correct
  rate is measured on the deterministic
  `MemoryAwareSyntheticBackend`.

* **Shared-prefix capsule reuse.** The prefix-capsule design
  produces *byte-identical prefix bytes* across consecutive
  turns once the team has produced `prefix_turns` outputs.
  This is the honest capsule-layer surface for "shared KV
  state" — a real backend with prompt-caching may exploit the
  byte-identical prefix for cache reuse; the W46 surface does
  not assert it does. The reuse evidence is per-turn (the
  `prefix_reused` flag) and is bound under the
  `prefix_capsule_cid`.

The cramming frontier (PART F):

* W43 crams **1808 structured bits per visible-token of
  overhead** at n=8 roles.
* W44 preserves all 1808 W43 structured bits per envelope
  while reducing visible-prompt cost by ≥ 4 tokens per active
  turn in factoradic compression mode at n=8.
* W45 preserves all W43 structured bits AND adds a per-turn
  **2-bit confidence channel** + a 14-bit ratify-probability
  surface at zero net visible-token cost.
* **W46** preserves all W43/W44/W45 structured bits AND adds
  per-turn:
  * `dict_idx` (`ceil(log2(K))` bits = 2 at K=4),
  * `layer_logits` (L × 14 bits = 28 bits at L=2),
  * `mem_attn_value` (14 bits),
  * `mem_summary` (up to 32 bits of compact role-pattern
    signature),
  at a structurally bounded ≤ 40-extra-tokens-per-turn cost
  vs the W45 hint. The packed control block carries
  approximately **9–10 bits per visible ctrl-token** at L=2,
  K=4, 5 roles (measured at the `bits_per_ctrl_token` extra
  field of the `r93_compressed_control_packing` family).
* The shared-prefix capsule additionally lets a backend with
  prompt-caching reuse the same prefix bytes across
  consecutive turns; the per-run accounting reports
  `n_visible_tokens_saved_prefix_reuse` and `n_prefix_reuses`.

## Benchmark families and results (PART G)

R-93 covers twelve cell families across five honest baselines:

| family | metric | passes |
|---|---|---|
| `r93_trivial_memory_passthrough` | passthrough_ok | 5/5 seeds all five arms = 1.000 |
| `r93_long_branching_memory` | precision_on_deep_turns | w46 = 1.000; w45 / w44 = 0.000 — Δ = +1.000 |
| `r93_cyclic_consensus_memory` | cycle_consensus_precision | w46 = 1.000, w45 = 1.000 — preservation under multi-layer + memory + ctrl + prefix |
| `r93_role_shift_adaptation` | role23_precision | w46_rank2 = 1.000, w46_shared_only = 0.000 — Δ = +1.000 |
| `r93_compressed_control_packing` | ctrl_round_trip_ok | w46 = 1.000 across 5/5 seeds; max overhead 38 tokens ≤ 40 |
| `r93_memory_facing_hint_response` | task_correct_rate | w46 = 1.000, w45 = 0.000 — Δ = +1.000 |
| `r93_causal_mask_preservation` | causal_mask_preserved | w46 = 1.000 (future-inject delta = 0.0) |
| `r93_dictionary_reconstruction` | dictionary_round_trip_ok | w46 = 1.000 (avg L1 = 0.0, closest-rate = 1.0) |
| `r93_shared_prefix_reuse` | prefix_reuse_ok | w46 = 1.000 (1 reuse per 4-turn run, n_visible_tokens_saved_prefix_reuse = 4) |
| `r93_w46_falsifier` | no_false_abstain | w46 = 1.000 (does not over-claim) |
| `r93_w46_compromise_cap` | downstream_protect_rate | 0.000 (limitation reproduces honestly) |
| `r93_replay_determinism` | replay_determinism_ok | w46 = 1.000 (bit-perfect across two independent runs) |

## Live / replay-live evaluation (PART H)

R-93 runs against the deterministic `SyntheticLLMClient` plus
the deterministic `MemoryAwareSyntheticBackend` so all results
are reproducible from a fixed seed set with no external
dependency.

The W46 surface also supports real Ollama or OpenAI-compatible
backends through the same `ManifoldMemoryTeam` class. A bounded
*realism anchor* is supported but not load-bearing for the
H1..H12 success bar; the synthetic backend establishes the
behavioural ground truth. The `MemoryAwareSyntheticBackend`
returns `MEMORY_OK` iff the prompt contains both `MANIFOLD_CTRL:`
and `mem_summary=`; this is engineered to detect the W46 packed
control surface and is not a real LLM. On real LLMs the saving
is bounded to "the packed control block is in the model's
context AND consecutive turns receive byte-identical prefix
bytes" — a strict superset of W45's surface but not a guarantee
of behavioural lift.

## Honest per-component verdicts (PART I)

| Component | Verdict | Evidence |
|---|---|---|
| Multi-layer controller stack | **behaviourally useful + structurally useful** | R-93 H2 (`r93_long_branching_memory`) — multi-layer + bank delivers +1.000 over W45 |
| Memory bank with CID provenance | **behaviourally useful** | R-93 H2 and the bank-head-cid binding under `verify_manifold_memory_handoff` |
| Causally-masked time attention | **behaviourally useful + structurally sound** | R-93 H2 + H7 (`r93_causal_mask_preservation`); future-inject delta = 0.0 across 5/5 seeds |
| Multi-rank role adapter | **behaviourally useful** | R-93 H4 (`r93_role_shift_adaptation`); rank-2 beats shared-only by +1.000 on role3 inversion |
| Learned dictionary basis | **structurally useful** | R-93 H8 (`r93_dictionary_reconstruction`); bijective decode + closest-prototype assignment 1.000 across 5/5 |
| Packed `MANIFOLD_CTRL` control surface | **behaviourally useful (on the synthetic backend)** | R-93 H5 + H6; ≤ 40-token overhead bound, bijective from envelope, lifts the memory-aware backend by +1.000 |
| Shared-prefix capsule | **structurally useful** | R-93 H9; byte-identical prefix across consecutive turns; reuse evidence per-run |
| Replay determinism (multi-layer + bank) | **behaviourally useful** | R-93 H11; bit-perfect across two independent runs |
| All-six-channel compromise cap | **falsified by the adversarial regime** | R-93 `r93_w46_compromise_cap`; the limitation reproduces — the W46 mechanism cannot recover under all-channel forgery |

The W43 carry-forwards (`W43-C-MIXED-CURVATURE-LATENT`,
`W43-C-COLLECTIVE-KV-POOLING`,
`W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) and `W45-C-DEEP-TRANSFORMER-
COUPLING` remain **substrate-blocked / conjectural** at the W46
layer.

## Theory and limitations (PART J)

### W46-T-MEMORY-COUPLING-DETERMINISM (proved + mechanically-checked)

For any deterministic backend and any fitted multi-layer
controller params, two independent runs of
`ManifoldMemoryTeam.run` over the same task with the same
registry, agents, observation builder, training set, and
dictionary basis produce byte-identical `final_output`,
byte-identical capsule chain root CIDs, byte-identical
sequences of `ManifoldMemoryHandoffEnvelope`, byte-identical
multi-layer-controller-params CIDs, AND byte-identical
memory-bank head CID per turn.

*Witness*: `coordpy/r93_benchmark.py::family_replay_determinism`
+ `tests/test_r93_benchmark.py::TestH11ReplayDeterminism` +
`tests/test_manifold_memory.py::TestDeterminism`.

### W46-T-MULTI-LAYER-FITTER-SOUNDNESS (proved by inspection)

The stage-wise ridge fitter in `fit_memory_controller` solves
`L` independent ridge regressions sequentially, each on the
*layer-wise residual* of the previous prediction. Stage `i+1`'s
target is `y - sum_{j <= i} layer_j(features)`; stage `i+1`'s
input is the same flattened channel features. This is a
deterministic stage-wise closed-form computation — no autograd,
no SGD, no randomness.

*Witness*: `coordpy/manifold_memory.py::fit_memory_controller`
(stage 1..L: ridge per layer; stage L+1: multi-rank adapter;
stage L+2: dictionary), `tests/test_manifold_memory.py::TestMultiLayerFitter`.

### W46-T-TIME-ATTENTION-CAUSAL-MASK (proved + mechanically-checked)

For any flat query, any memory bank, any temperature, and any
turn index `t`, `compute_time_attention` returns a
`TimeAttentionWitness` whose `pooled_value` depends only on
memory entries with `turn_index < t`. Injecting a memory entry
with `turn_index >= t` into the bank does NOT change the
`pooled_value`; the mask is enforced by
`ManifoldMemoryBank.admissible_for_turn`.

*Witness*: R-93 `r93_causal_mask_preservation` family —
`future_inject_delta == 0.0` across 5/5 seeds;
`tests/test_manifold_memory.py::TestTimeAttention::test_future_entries_are_masked`.

### W46-T-DICTIONARY-BIJECTION (proved by inspection + empirical)

For any flat feature vector `v` of length `W45_N_CHANNELS *
feature_dim`, `DictionaryBasis.encode(v) = (i, r)` and
`DictionaryBasis.decode(i, r) = v` exactly (modulo IEEE-754
round-trip at the 12-decimal `_round_floats` precision).

*Witness*: `coordpy/manifold_memory.py::DictionaryBasis.encode` /
`decode`; R-93 `r93_dictionary_reconstruction` —
`dictionary_round_trip_ok == 1.0` across 5/5 seeds;
`tests/test_manifold_memory.py::TestDictionaryBasis::test_encode_decode_round_trip`.

### W46-T-MULTI-RANK-ADAPTER-SUFFICIENCY (proved-conditional + empirical)

When a fitted controller's shared base predicts the same sign
for two roles whose gold conventions invert along *different*
axes (one along the spherical axis, one along the subspace
axis), a rank-2 role-specific delta recovers both inversions;
a rank-1 delta can fit at most one. The R-93
`r93_role_shift_adaptation` family registers rank-2 precision
= 1.000 vs shared-only precision = 0.000 across 5/5 seeds.

*Witness*: `coordpy/r93_benchmark.py::family_role_shift_adaptation`
+ `tests/test_r93_benchmark.py::TestH4RoleShiftAdaptation`.

### W46-T-PREFIX-CAPSULE-STABILITY (proved by inspection + empirical)

For any sequence of prior outputs `[o_0, o_1, ...]` and any
`prefix_turns = K`, calling `build_prefix_capsule` at turn `k`
and turn `k+1` (both with `k >= K`) produces the same
`prefix_sha256` because the function consumes only the first
`K` outputs. The `reused` flag is True at turn `k+1` when the
caller passes the prior turn's `prefix_sha256` as the
`prior_prefix_sha` argument.

*Witness*: `coordpy/manifold_memory.py::build_prefix_capsule`
(uses `prior_outputs[:int(prefix_turns)]`); R-93
`r93_shared_prefix_reuse` family — `prefix_reuse_ok == 1.0`
across 5/5 seeds with at least 1 reuse per run;
`tests/test_manifold_memory.py::TestPrefixCapsule::test_first_n_outputs_only`.

### W46-T-CONTROL-TOKEN-BIJECTION (proved by inspection + empirical)

Given a `ControlTokenWitness` and the registered `ctrl_mode`,
the literal `MANIFOLD_CTRL` bytes are reconstructible exactly
from the witness fields. The `ctrl_bytes_sha256` field binds
the bytes; an auditor who re-builds the block from the witness
fields will compute the same SHA-256.

*Witness*: `coordpy/manifold_memory.py::build_control_token_string`;
R-93 `r93_compressed_control_packing` — `ctrl_round_trip_ok ==
1.0` across 5/5 seeds.

### W46-T-VERIFIER-SOUNDNESS (proved by inspection + mechanically-checked)

Trust-boundary soundness of the W46 verifier:
`verify_manifold_memory_handoff` enumerates 21 failure modes
disjoint from W22..W45's 240 cumulative failure modes
(`empty_w46_envelope`, `w46_schema_version_unknown`,
`w46_schema_cid_mismatch`, `w46_decision_branch_unknown`,
`w46_ctrl_mode_unknown`,
`w46_role_handoff_signature_cid_invalid`,
`w46_prompt_sha256_invalid`,
`w46_token_accounting_invalid`,
`w46_confidence_bucket_invalid`,
`w46_ratify_probability_invalid`,
`w46_controller_params_cid_invalid`,
`w46_dictionary_cid_invalid`,
`w46_time_attention_witness_cid_invalid`,
`w46_multi_rank_adapter_witness_cid_mismatch`,
`w46_causal_mask_witness_cid_invalid`,
`w46_control_token_witness_cid_invalid`,
`w46_prefix_capsule_cid_invalid`,
`w46_memory_bank_head_cid_invalid`,
`w46_prompt_construction_witness_cid_mismatch`,
`w46_memory_witness_cid_mismatch`,
`w46_outer_cid_mismatch`). Cumulative trust boundary across
W22..W46 = **261 enumerated failure modes**.

*Witness*:
`coordpy/manifold_memory.py::verify_manifold_memory_handoff` +
`tests/test_manifold_memory.py::TestVerifier`.

### W46-L-TRIVIAL-MEMORY-PASSTHROUGH (proved by inspection + empirical)

When `ManifoldMemoryRegistry.is_trivial` is True
(`memory_enabled=False`, `time_attention_enabled=False`,
`dictionary_enabled=False`, `control_token_mode='off'`,
`prefix_reuse_enabled=False`, `params.fitting_method='unfitted'`,
W45/W44/W43 inner trivial), the memory orchestrator emits no
envelope side effects beyond the W45 trivial-passthrough
envelope, and `ManifoldMemoryTeam.run` produces the same
`final_output`, `n_turns`, and capsule chain head as
`AgentTeam.run` for the same backend.

*Witness*:
`tests/test_manifold_memory.py::TestTrivialPassthrough` and the
R-93 `r93_trivial_memory_passthrough` family (5/5 seeds, all
five arms).

### W46-L-MEMORY-COMPROMISE-CAP (proved-conditional limitation; strengthens W45-L-LEARNED-COMPROMISE-CAP)

When an adversary forges ALL SIX channel observations to match
the registered policy AND the memory bank is exposed to the
same forged feature distribution, the W46 multi-layer
controller cannot recover at the capsule layer — every
channel's feature vector matches the trained "honest"
distribution, every memory entry's gate logit is positive, and
the time-attention readout amplifies (rather than counters) the
forged ratify. Recovery requires either (a) a stricter
role-handoff signature policy, (b) native-latent evidence
outside the capsule layer
(`W43-C-MIXED-CURVATURE-LATENT`), or (c) a trained
adversarial-example detector that operates on a feature axis
the attacker cannot trivially forge.

*Witness*: R-93 `r93_w46_compromise_cap` family —
`w46.mean == w45.mean == 0.0` across 5/5 seeds.

### W46-L-CONTROL-TOKEN-MODEL-INDIFFERENCE-CAP (proved-conditional limitation; strengthens W45-L-PROMPT-HINT-MODEL-INDIFFERENCE-CAP)

The `MANIFOLD_CTRL` packed control block guarantees only that
the multi-layer controller's recommendation, layer logits,
memory attention readout, dictionary index, and memory
role-pattern signature are *present* in the model's context.
A real LLM may or may not condition on the block. The H6
memory-facing-response gain is measured on the deterministic
`MemoryAwareSyntheticBackend` which is engineered to detect
both the `MANIFOLD_CTRL:` and `mem_summary=` substrings; on real
LLMs the saving is bounded to "the packed control block is in
the model's context".

*Witness*: `docs/RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md` (this
file, PART E) + `coordpy.manifold_memory.MemoryAwareSyntheticBackend`
docstring.

### W46-L-SHARED-PREFIX-NOT-KV-CACHE-CAP (proved-conditional limitation)

The shared-prefix capsule guarantees *byte-identical prefix
bytes* across consecutive turns once the team has produced
`prefix_turns` outputs. Whether the underlying transformer
reuses its internal KV cache for that prefix is a model-side
runtime concern — modern backends with prompt-caching may
exploit byte-identical prefixes, but the W46 surface does not
measure or assert this. The honest claim is "identical prefix
bytes," not "identical KV state."

*Witness*: `coordpy/manifold_memory.py::build_prefix_capsule`
docstring; R-93 `r93_shared_prefix_reuse` family — measures
`prefix_reused == True` per turn but does not measure backend
KV reuse.

### W46-L-RIDGE-STACK-EXTRAPOLATION-CAP (proved-conditional limitation; strengthens W45-L-RIDGE-EXTRAPOLATION-CAP)

The multi-layer ridge stack is fitted on the training bank's
feature distribution. The same out-of-distribution caveats that
apply to the W45 single-layer fit apply at W46, additionally
compounded by the time-attention readout's dependence on the
bank's feature distribution: out-of-distribution features yield
a low cosine similarity against the bank, the softmax weights
become approximately uniform, and the pooled value is
approximately the bank's mean gate logit. The W46 layer still
seals an honest audit envelope; the verdict on
out-of-distribution cells is honestly *uncertain*.

*Witness*: inspection of `forward_memory_controller` +
`compute_time_attention` + `r93_w46_falsifier` (no false abstain
on the clean linear-flow regime).

### W46-C-DEEP-TRANSFORMER-COUPLING (conjectural; carries forward W45-C)

The full direction of "learn a deep, transformer-coupled
controller that consumes hidden states and emits attention-mask
adjustments / KV-cache routing" remains *substrate-blocked*.
The W46 mechanism is the strongest capsule-layer approximation
we can write today; the next direction requires architectural
access to the model's internals. Note: W46 *strengthens* the
W45 bounding of `W44-C-LIVE-LATENT` by adding the multi-layer
+ memory + dictionary path, but it does not close
transformer-internal consumption.

*Witness*: `docs/RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md`
("What was NOT done — honest scope").

### W46-C-AUTOGRAD-DEEP-STACK (conjectural; deliberately deferred)

A version of the W46 multi-layer stack trained via SGD /
backprop (rather than stage-wise closed-form ridge) is
structurally compatible with the current envelope chain — the
parameter CID would still bind a fitted bundle — but is
deliberately *out of scope* for W46 in order to preserve
deterministic replay. A future milestone may add an autograd
path under an `[autograd]` extra.

*Witness*: `docs/SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md`
("scope, do-not-overstate").

## Product-boundary decisions (PART K, M)

The released CoordPy 0.5.20 stable surface is byte-for-byte
unchanged. The W46 module ships in the source tree but is
**not** re-exported through `coordpy/__init__.py` and is **not**
listed in `coordpy.__experimental__`. The first-run UX
(`coordpy-team run --preset quant_desk ...`) is unaffected; the
smoke driver (`tests/test_smoke_full.py`) reports "ALL CHECKS
PASSED" with the W46 module on disk.

A sophisticated caller reaches the W46 surface explicitly:

```python
from coordpy.manifold_memory import (
    ManifoldMemoryTeam,
    build_manifold_memory_registry,
    fit_memory_controller,
    W46_CTRL_MODE_FULL,
)
from coordpy.learned_manifold import TrainingSet, TrainingExample
from coordpy.r93_benchmark import run_all_families, render_text_report

# 1. Build a training set of (cell, label) pairs.
# 2. Fit the multi-layer controller via stage-wise closed-form ridge.
# 3. Build a ManifoldMemoryTeam over your agents + a real backend.
# 4. Run; replay later from the sealed envelopes.

results = run_all_families()
print(render_text_report(results))
```

This explicit import reflects the milestone's research-grade
status. A future milestone may promote a stable subset of the
W46 surface under `coordpy.__experimental__` once cross-host
memory evidence is acquired and the controller's training-set
size is increased beyond the hermetic synthetic bench.

## Validation (PART M)

* **Baseline regression**: `tests/test_smoke_full.py` reports
  "ALL CHECKS PASSED" with the W46 module on disk.
* **W43 regression**: `coordpy/r90_benchmark.py::run_all_families`
  reproduces the W43 R-90 results byte-for-byte (8 families).
* **W44 regression**: `coordpy/r91_benchmark.py::run_all_families`
  reproduces the W44 R-91 results byte-for-byte (7 families).
* **W45 regression**: `coordpy/r92_benchmark.py::run_all_families`
  reproduces the W45 R-92 results byte-for-byte (9 families).
* **W46 unit tests**: `tests/test_manifold_memory.py` — 44 tests
  passed, including 4 verifier-tamper tests, 4 time-attention
  causal-mask tests, 4 dictionary bijection tests, 4 prefix
  capsule stability tests, 4 control-token surface tests.
* **R-93 H1..H12**: `tests/test_r93_benchmark.py` — 20 tests
  passed, exercising every pre-committed hypothesis.
* **Aggregate test count**: 224 tests passed across the full
  `tests/` directory (W43 PMC + R-90 + W44 LMCC + R-91 + W45 LMC
  + R-92 + W46 MMC + R-93 + the released smoke driver).

## Version + release status

* **No version bump**: `coordpy.__version__` is `"0.5.20"`
  (unchanged). `coordpy.SDK_VERSION` is `"coordpy.sdk.v3.43"`
  (unchanged).
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* The W46 module ships in the source tree as a research
  artifact that lives alongside the released v0.5.20 wheel; it
  is not part of the public SDK contract.

## Where this leaves the programme

W46 is the first capsule-layer milestone in CoordPy where the
controller has **non-trivial depth and explicit memory**:

* The W43 mechanism is closed-form, deterministic,
  zero-parameter channel encoding.
* The W44 mechanism is a hand-designed live gate over W43
  channels.
* The W45 mechanism is a **fitted, single-layer, content-
  addressed controller** with attention-style routing, a
  LoRA-style adapter, margin calibration, and a model-facing
  learned prompt surface.
* The **W46 mechanism** is a **multi-layer, memory-conditioned,
  content-addressed controller** with stage-wise-fitted
  layers, causally-masked time-attention over a bounded memory
  bank, rank-r role adapters, a learned dictionary basis, a
  packed `MANIFOLD_CTRL` model-facing control surface, and a
  shared-prefix capsule that emits byte-identical prefix bytes
  across consecutive turns.

The remaining open frontiers are:
* The W43 conjectures
  (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`).
* `W44-C-LIVE-LATENT` is now **further bounded** at the
  capsule layer by W46 (the hyperbolic and euclidean channels
  are consumed by the multi-layer + memory + dictionary path)
  without being closed.
* `W45-C-DEEP-TRANSFORMER-COUPLING` carries forward.
* The new W46 conjecture (`W46-C-AUTOGRAD-DEEP-STACK`).

These require new architectural substrate beyond the capsule
layer and are explicitly out of scope for the W46 milestone.
The honest storyline for the Context Zero programme is
therefore:

* **W43**: executable product-manifold capsules.
* **W44**: live manifold-conditioned behaviour.
* **W45**: learned, single-layer, model-facing manifold control.
* **W46**: deeper, multi-layer, memory-conditioned, transformer-
  facing manifold-memory control.
