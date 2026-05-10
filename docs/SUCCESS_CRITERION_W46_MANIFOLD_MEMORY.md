# Pre-committed success criterion — W46 Manifold Memory Controller

> Programme step: post-W45. Mints axis 43 of the Context Zero
> programme. Strictly additive on top of W45 LMC, W44 LMCC, W43 PMC,
> and the released v3.43 line. Honest scope: the W46 mechanism
> replaces the *single-layer*, *per-cell*, *prompt-line*
> learned controller of W45 with a **multi-layer learned
> manifold memory controller** that maintains a bounded,
> content-addressed memory of previous-turn channel features,
> performs causally-masked time-attention reads from that bank,
> stacks rank-r role adapters across layers, encodes channel
> state into a learned dictionary basis, and emits a packed,
> multi-token, content-addressed `MANIFOLD_CTRL` control surface
> plus a deterministic shared-prefix capsule. It does NOT close
> the W43 conjectures (`W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`), does NOT promote the
> `W45-C-DEEP-TRANSFORMER-COUPLING` carry-forward to a closed
> claim, and does NOT manipulate transformer hidden state, KV
> cache, attention weights, or embeddings. The learned memory
> controller is an executable proxy at the capsule layer.

## Mechanism

W46 introduces the **Manifold Memory Controller (MMC)** — the
first capsule-native CoordPy layer where the gating policy is
shaped by **deeper, multi-layer, memory-conditioned learned
structure** with explicit causal masking across time, roles, and
channels. The mechanism is composed of seven learned and
content-addressed components, all closed-form-fittable in pure
Python / stdlib so they can be trained deterministically without
an autograd dependency:

* **Multi-layer learned controller stack.** A stack of `L` layers
  applied to the per-channel feature vectors. Each layer applies
  a fitted projection + softmax attention pool + residual add.
  The W45 single-layer controller is the `L=1` degenerate case;
  W46 fits `L >= 2` via stage-wise closed-form ridge on the
  layer-wise residuals. The shared base is consequently a
  deeper, content-addressed function of the channel features
  than the W45 single layer.

* **Manifold memory bank.** A bounded, content-addressed ring
  buffer of the last `M` turn observations: each entry stores
  the turn index, role, role-handoff-signature CID, per-channel
  feature vector, per-channel logits, gate logit, ratify
  probability, and decision branch. The bank's CID is the
  SHA-256 over its sorted entries; an auditor can re-derive the
  bank state at any turn from the envelope chain alone.

* **Causally-masked time-attention.** When ratifying turn `t`,
  the controller computes attention over memory entries `< t`
  using a learned cosine-similarity readout: the per-channel
  features of turn `t` query the bank, the bank's gate logits
  are the values, and the pooled output is the *time-attention
  context* — a single scalar added to the layer-wise gate logit
  before the margin sigmoid. The mask is strict: a turn cannot
  attend to itself or to the future.

* **Multi-rank role adapter stack.** The W45 rank-1 LoRA-style
  adapter is generalized to rank `r >= 1`. The per-role delta
  carries `r * n_channels + 1` scalars instead of `r + 1`; the
  basis vectors are the *signed per-channel logits* (basis 0)
  plus their rotations (bases 1..r-1). Each per-role delta is
  fit via closed-form ridge regression on the per-role
  residuals; the bound on the residual norm is reported in the
  audit envelope.

* **Learned dictionary basis.** A small dictionary of `K`
  prototype channel-feature vectors, fitted across the training
  bank via deterministic K-prototype assignment (closest
  prototype gets the example). Per-turn, the controller encodes
  the cell's channel features as the index of the closest
  prototype + a small residual; this is a **sparse code** with a
  bijective round-trip from `(index, residual)`. The dictionary
  CID + per-turn index + residual norm are part of the envelope.

* **Packed control token surface (`MANIFOLD_CTRL`).** Replaces
  the W45 single-line hint with a structured multi-line control
  block:

  ```
  MANIFOLD_CTRL:
    route=<int>
    conf=<bucket>
    p=<rounded probability>
    layer_logits=[<l1>,<l2>,...,<lL>]
    mem_attn=<rounded scalar>
    dict_idx=<int>
    mem_summary=<role-pattern-signature>
    over <role universe>
  ```

  Strictly more *structured bits per visible-token* than W45's
  single-line hint at the same overall token budget. The bytes
  are content-addressed and bijectively recoverable from the
  envelope.

* **Shared-prefix capsule.** A deterministic prefix derived from
  the last `P` turn-output SHAs and the registered policy entry.
  When `prefix_reuse_enabled=True`, the prompt builder reuses
  the same prefix bytes across consecutive turns and the
  envelope binds the prefix CID. This is an honest capsule-layer
  approximation of "shared KV state" — no transformer-internal
  KV cache is touched; the model sees an identical prefix byte
  sequence across turns. The prefix CID becomes part of the
  envelope chain so an auditor can prove "these N turns received
  the same prefix bytes."

The W46 layer is strictly additive on top of W45 and the
released v3.43 SDK. When the memory controller is configured
trivially (`memory_enabled=False`, `n_layers=1`,
`time_attention_enabled=False`, `dictionary_enabled=False`,
`control_token_mode='off'`, `prefix_reuse_enabled=False`, role
adapter rank `r=1`, W45 inner trivial), the W46 orchestrator
reduces to `LearnedManifoldTeam.run` byte-for-byte — the
**W46-L-TRIVIAL-MEMORY-PASSTHROUGH** falsifier.

## Pre-committed hypotheses (H1..H12)

Each hypothesis is testable from the bundled R-93 benchmark
family (`coordpy.r93_benchmark`) and its accompanying pytest
tests (`tests/test_r93_benchmark.py`). Pre-committed seed set:
`(0, 1, 2, 3, 4)`. Backend: deterministic `SyntheticLLMClient`
plus `MemoryAwareSyntheticBackend` for the memory-conditioned
families; a real Ollama backend is supported through the same
surface but is not load-bearing for the pre-committed bar.

* **H1 — Trivial memory passthrough is byte-for-W45.**
  Family `r93_trivial_memory_passthrough`: with the zero-fitted
  memory registry, `ManifoldMemoryTeam.run` produces the same
  `final_output`, `turns`, and capsule chain head as
  `LearnedManifoldTeam.run` for the same backend.
  **Pass if**: `passthrough_ok = 1.0` across 5/5 seeds.

* **H2 — Long branching memory gain strictly beats W45 on the
  multi-turn-context regime.**
  Family `r93_long_branching_memory`: a regime where the gating
  decision at turn `t` depends on the *sequence* of prior gate
  logits (e.g., a branching pattern that closes only when the
  history matches a registered branch). W45's single-layer
  controller has no memory and cannot resolve the branch; W46's
  time-attention can.
  **Pass if**: `w46.mean - w45.mean >= 0.20` precision gain AND
  `w46.min >= 0.8` across 5/5 seeds.

* **H3 — Cyclic-consensus memory tracking (preservation under
  multi-layer + memory).**
  Family `r93_cyclic_consensus_memory`: a 4-cycle regime where
  each role must produce the consensus claim_kind for the cycle
  phase. The W45 hand-tuned spherical gate already hits 1.0 on
  this regime when configured with a moderate threshold; the
  W46 multi-layer + memory + control-token + prefix path must
  *preserve* that ceiling without regressing.
  Honest scope: this is a preservation check, not a strict
  beat. The W46 layer adds richer state (memory bank +
  multi-layer + control surface) on top; the bar is that the
  added machinery does not perturb the cyclic-consensus
  accuracy.
  **Pass if**: `w46.min == w46.max == 1.0` AND `w46.mean >=
  w45.mean` across 5/5 seeds.

* **H4 — Multi-rank role adapter strictly beats rank-1 on a
  multi-axis role-shift regime.**
  Family `r93_role_shift_adaptation`: a 4-role team where two
  roles have orthogonal sign conventions (one flipped on the
  spherical axis, one flipped on the subspace axis). A rank-1
  delta can fit at most one of the two; rank-2 can fit both.
  **Pass if**: `w46_rank2.mean - w46_rank1.mean >= 0.20` AND
  `w46_rank2.min == 1.0` across 5/5 seeds.

* **H5 — Packed control token surface adds bits-per-token over
  W45 hint without changing visible-token cost class.**
  Family `r93_compressed_control_packing`: with
  `control_token_mode='full'`, the prompt's `MANIFOLD_CTRL`
  block carries `layer_logits + mem_attn + dict_idx +
  mem_summary` in addition to W45's `route + conf + p`. The
  bytes are bijectively recoverable from the envelope; the
  visible-token count grows by a *structurally bounded
  constant* per turn (one short multi-line YAML block,
  ≤ 40 extra tokens, independent of team size beyond layer
  count) compared with W45's hint.
  **Pass if**: control_token_round_trip = 1.0 across 5/5 seeds
  AND per-turn extra tokens vs W45 ≤ 40.

* **H6 — Memory-facing hint response: memory-aware backend lifts
  task-correct rate.**
  Family `r93_memory_facing_hint_response`: a deterministic
  `MemoryAwareSyntheticBackend` that returns one canonical
  answer when the prompt contains `MANIFOLD_CTRL:` *and*
  `mem_summary=` substrings, and a different answer otherwise.
  W46 arm strictly raises the task-correct rate.
  **Pass if**: `w46.mean - w45.mean >= 0.40` across 5/5 seeds.

* **H7 — Causal mask preservation on the memory bank.**
  Family `r93_causal_mask_preservation`: at turn `t`, the
  controller's time-attention read must not be influenced by
  any memory entry with index `>= t`. A regression probe
  injects a future-turn entry into the bank and asserts the
  pooled output is unchanged.
  **Pass if**: future-injection delta = 0.0 across 5/5 seeds.

* **H8 — Dictionary reconstruction is bijective.**
  Family `r93_dictionary_reconstruction`: the per-turn
  `(dict_idx, residual)` pair is sufficient to recover the
  channel features modulo `||residual||_inf` precision. We
  measure the L1 reconstruction error vs the original features.
  **Pass if**: reconstruction L1 ≤ residual L1 (i.e., the
  dictionary basis explains the non-residual mass) AND
  `dict_idx` matches the closest prototype for every cell.

* **H9 — Shared-prefix capsule reuses bytes across consecutive
  turns AND saves visible tokens.**
  Family `r93_shared_prefix_reuse`: with `prefix_reuse_enabled=
  True`, the prompt bytes' SHA-256 across turns 1..T-1 contains
  a stable prefix substring (the same first K bytes) AND the
  per-turn visible-token cost is strictly less than W45's at the
  same agent count.
  **Pass if**: prefix_stable_across_turns = 1.0 AND
  `w46.mean_tokens - w45.mean_tokens <= -1.0` (at least 1
  visible token saved per turn) across 5/5 seeds.

* **H10 — W46 envelope verifier enumerates ≥ 16 disjoint failure
  modes.**
  `verify_manifold_memory_handoff` returns the empty-envelope
  failure as the first mode and detects tampering with the
  controller-stack params CID, time-attention witness CID,
  multi-rank-adapter witness CID, dictionary CID, memory-bank
  CID, control-token witness CID, prefix-capsule CID, and outer
  CID through one of the disjoint named modes.
  **Pass if**: a successful verification reports `n_checks >=
  16` AND `len(W46_ALL_FAILURE_MODES) >= 16` AND the modes are
  disjoint from W22..W45's named modes (cumulative trust
  boundary ≥ 256 enumerated failure modes).

* **H11 — Replay determinism: bit-perfect replay of a memory-
  controller run.**
  For any deterministic backend, two independent
  `ManifoldMemoryTeam.run` invocations over the same task with
  the same registry, agents, observation builder, fitted
  controller params, and dictionary basis produce byte-identical
  `final_output`, byte-identical capsule chain root CIDs,
  byte-identical sequences of `ManifoldMemoryHandoffEnvelope`,
  byte-identical multi-layer-controller-params CID, AND
  byte-identical memory-bank head CID per turn.
  **Pass if**: `r93_replay_determinism` passes across 5/5 seeds.

* **H12 — Stable SDK contract preserved.**
  CoordPy 0.5.20's stable smoke driver
  (`tests/test_smoke_full.py`) reports "ALL CHECKS PASSED" with
  the W46 module on disk. The W46 surface is held outside
  `coordpy.__experimental__`; it is reachable only through an
  explicit `from coordpy.manifold_memory import ...` import. The
  released v0.5.20 wheel's public surface is byte-for-byte
  unchanged.
  **Pass if**: smoke driver reports "ALL CHECKS PASSED" with
  the W46 module installed AND `coordpy.__version__ == "0.5.20"`
  AND `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`.

## Falsifiers + per-channel verdicts

Each W43 channel is force-verdicted at the *memory* layer:

| Channel             | W45 verdict                    | W46 verdict                                |
|---------------------|--------------------------------|--------------------------------------------|
| Spherical consensus | learned-margin gate            | + multi-layer; + memory-attended; + rank-r |
| Subspace drift      | learned-margin gate            | + multi-layer; + memory-attended; + rank-r |
| Causal clock        | learned-margin gate            | + memory-attended (time mask)              |
| Factoradic route    | learned + confidence-bucketed  | + packed control token + mem_summary       |
| Hyperbolic branch   | learned feature                | + dictionary-coded + memory-attended       |
| Euclidean attribute | learned feature                | + dictionary-coded + memory-attended       |

This is the structural lift over W45: every channel now flows
through the memory bank's time attention AND the multi-layer
controller stack AND the learned dictionary, AND every channel
contributes to the packed control token surface. The
**`W45-C-DEEP-TRANSFORMER-COUPLING`** carry-forward is *bounded*
(not closed) at the capsule layer: the controller is now
genuinely multi-layer with cross-time attention, but it remains
a capsule-layer proxy that does not touch transformer-internal
state.

A channel's W46 verdict fails if (a) its weight in the trained
controller stack is identically zero across every layer, or (b)
the corresponding R-93 family does not register the predicted
gain. The hyperbolic and euclidean channels are required to
register a non-zero dictionary code and a non-zero memory
attention readout on at least one signature in the bench.

## Scope (do-not-overstate)

W46 does NOT:

* claim transformer-internal access. The memory controller
  operates strictly over W43 capsule-layer channel encodings; it
  does not read hidden states, transplant KV cache, inspect
  attention weights, or modify the model's attention
  computation. The W43 conjectures
  (`W43-C-MIXED-CURVATURE-LATENT`,
  `W43-C-COLLECTIVE-KV-POOLING`,
  `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`) and
  `W45-C-DEEP-TRANSFORMER-COUPLING` carry forward unchanged.

* claim the multi-layer controller is a deep neural network in
  the autograd sense. Each layer is fitted closed-form via stage
  -wise ridge on layer-wise residuals; there is no SGD, no
  backprop, no learned non-linearity beyond the per-layer
  softmax attention pool. It is a deterministic, stage-wise
  fitted *stack* of capsule-layer projections.

* claim true shared-KV between turns. The shared-prefix capsule
  guarantees byte-identical prefix bytes across turns; whether
  the underlying transformer reuses its internal KV cache for
  that prefix is a model-side runtime concern that the W46
  surface does not measure or assert. The honest claim is
  "identical prefix bytes" not "identical KV state."

* claim true time-attention on the model's hidden state. The
  W46 time-attention is a cosine-similarity readout over
  capsule-layer channel features; the model's attention
  computation is unchanged.

* claim the multi-rank role adapter is true rank-r LoRA on a
  transformer. The W46 multi-rank adapter is a rank-r
  perturbation of the capsule-layer policy parameters, not the
  model's weight matrices.

* claim multi-host learned coupling. The W46 surface is a
  same-process orchestrator; cross-host learning requires
  substrate beyond the capsule layer.

* claim the dictionary basis is a learned vocabulary in the
  language-modeling sense. It is a deterministic
  K-prototype clustering of the training-bank channel
  features.

## Strong / partial / failure

* **Strong success**: H1..H12 all pass; every W43 channel
  flows through the multi-layer + memory + dictionary path with
  measurable contribution; the released SDK contract is
  preserved; the cumulative trust-boundary count is at least
  256.

* **Partial success**: H1 + H7 + H10 + H11 + H12 pass plus at
  least three of H2..H6, H8..H9 pass; the rest are honestly
  downgraded with a per-component retraction.

* **Failure**: H1 or H7 or H10 or H11 or H12 regresses; the
  memory mechanism is rolled back behind the trivial registry
  while the W43/W44/W45 layers remain green.

## Stable boundary preservation (PART K)

The W46 surface ships at `coordpy.manifold_memory` and
`coordpy.r93_benchmark` and is reachable only through an
explicit import. The first-run UX (`coordpy-team run --preset
quant_desk ...`) is unaffected. The W43 surface
(`coordpy.product_manifold`, `coordpy.r90_benchmark`), the W44
surface (`coordpy.live_manifold`, `coordpy.r91_benchmark`), and
the W45 surface (`coordpy.learned_manifold`,
`coordpy.r92_benchmark`) are unchanged. The released
`AgentTeam.run`, `LiveManifoldTeam.run`, and
`LearnedManifoldTeam.run` paths are byte-for-byte unchanged —
`ManifoldMemoryTeam` is a new class that sits beside them. A
future milestone may promote a stable subset of the W46 surface
under `coordpy.__experimental__` once cross-host memory evidence
is acquired.

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`
  for this milestone. `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* The W46 module ships in the source tree as a research
  artifact that lives alongside the released v0.5.20 wheel; it
  is not part of the public SDK contract and is held outside
  `coordpy.__experimental__` for this milestone.

## Why W46 is the strongest honest next step

W45 made *one* learned gating decision per turn from *one*
fitted projection over a *flat* feature vector. The next honest
move is to let those decisions *depend on what the team has
done so far*, *stack across multiple fitted layers*, and *bind
a richer model-facing control surface* — but only as far as the
capsule layer permits.

The W46 mechanism is the smallest move that is genuinely:

* **multi-layer** (closed-form-stacked controller),
* **memory-conditioned** (bounded ring buffer of past channel
  features + gate logits, with causally-masked time attention),
* **multi-rank** (LoRA-style rank-r role adapters),
* **dictionary-coded** (sparse code over a learned K-prototype
  dictionary),
* **packed-control** (multi-token `MANIFOLD_CTRL` surface),
* **shared-prefix-aware** (deterministic prefix-capsule reuse),

while staying inside the audit boundary. We do not claim more
than that. The deeper architectural directions
(transformer-internal mixed-curvature attention, KV pooling,
continuous Grassmannian homotopy, real shared-state across
hosts, end-to-end backprop) require substrate beyond the
capsule layer and remain conjectural.
