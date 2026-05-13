# Success criterion — W56 Substrate-Coupled Latent Operating System (SCLOS)

> Pre-committed, falsifiable bar for the post-W55 W56 milestone
> (SCLOS — first substrate-attack milestone in the Context Zero
> programme). 2026-05-13.
>
> If any H-bar fails on the released-bench at 3 seeds, that bar is
> **falsified** and the corresponding ``W56-L-…`` cap is recorded
> in `docs/THEOREM_REGISTRY.md`.

## TL;DR

W56 is the first milestone where the Context Zero latent operating
system actually couples to a **real, executable transformer
substrate** — token embeddings, multi-head self-attention, KV
cache, hidden states, layer norm, position-wise feed-forward,
logits — instead of operating only over capsule-layer proxies.
The substrate is a **small in-repo transformer research runtime**
(``coordpy.tiny_substrate``) implemented in pure NumPy plus a
deterministic LCG; we do NOT pretend that this is a frontier
production model and we do NOT pretend that we have breached
third-party transformer-internal access. We DO claim that — within
the bounded universe of this tiny runtime — KV bytes, hidden
state tensors, attention weights, and logits are genuinely real,
not metaphorical.

The previous milestones (W43..W55) built an increasingly strong
capsule-layer **proxy** stack. W56 retains all of that and **adds
a real substrate floor**: every load-bearing W56 mechanism either
reads from or writes to the tiny-substrate KV cache / hidden
state / logits, *plus* honestly reports what remains substrate-
blocked on third-party hosted backends (Ollama, OpenAI-compatible).

Twelve orthogonal mechanism advances on top of W55:

* **M1 Tiny Transformer Substrate** — ``coordpy.tiny_substrate``:
  a real, executable, content-addressed, deterministic
  small-vocab small-depth transformer with byte-level token
  embeddings, ``H``-head causal self-attention, per-layer KV
  cache, post-attention layer norm, position-wise feed-forward
  block, residual stream, unembedding head. Pure NumPy. No
  external model files. Hidden states / KV cache / attention
  weights / logits are observable and modifiable.
* **M2 Substrate Adapter** — ``coordpy.substrate_adapter``:
  honestly probes a backend for substrate-coupling capability
  (logits / hidden state / KV / attention). Returns a per-axis
  capability matrix and records the verdict in a content-
  addressed witness. The matrix discriminates between three
  honest tiers: ``substrate_full`` (tiny runtime), ``logits_only``
  (in principle Ollama with logprobs and embeddings), and
  ``text_only`` (most hosted APIs).
* **M3 KV Bridge** — ``coordpy.kv_bridge``: bridges capsule-layer
  latent state into the tiny substrate's KV cache. The bridge
  takes a latent carrier (e.g. an MLSC V4 capsule payload) and
  writes it into one or more reserved attention slots in the
  substrate's per-layer KV bank, then runs decode/forward. This
  is the load-bearing test of "does a capsule carrier actually
  change the model's output when injected as KV?". The answer is
  measured, not asserted.
* **M4 Persistent Latent State V8** — 6-layer V8 stacked cell
  with a fourth persistent skip-link (substrate-conditioned
  EMA), ``max_chain_walk_depth = 256``, and a substrate-aware
  merge head that consumes the tiny-substrate hidden state at
  merge time.
* **M5 Multi-Hop Translator V6** — 8-backend (A..H) over 56
  directed edges with chain-length-7 transitivity scoring,
  substrate-trust arbitration: per-backend trust derived from
  measured substrate fidelity (cosine to tiny-runtime hidden
  state) rather than declared trust.
* **M6 Mergeable Latent State Capsule V4 (MLSC V4)** — extends
  MLSC V3 with: (a) ``substrate_witness`` field holding the
  tiny-substrate hidden state hash at merge time; (b) per-fact
  ``provenance_chain`` walking back through merge ancestors;
  (c) ``algebra_signature`` recording which disagreement-algebra
  primitive (⊕/⊖/⊗) produced the merge.
* **M7 Consensus Fallback Controller V2** — extends TWCC with a
  6-stage decision chain {K-of-N → trust-weighted → substrate-
  conditioned → best-parent → transcript → abstain}: substrate-
  conditioned stage uses the tiny-runtime forward as a tiebreaker
  oracle when capsule consensus is split.
* **M8 Corruption-Robust Carrier V4** — composes BCH(31,16)
  triple-bit correction per segment + 7-of-9 majority repetition
  + 2D interleaving (rows + columns) so burst errors of length
  ≤ 5 are recoverable.
* **M9 Deep Substrate Hybrid Stack** — ``coordpy.deep_substrate_hybrid``:
  replaces the top ``L_out`` layers of Deep Proxy Stack V6 with
  the tiny-substrate attention block. Reads / writes the real
  KV cache. Provides the first load-bearing "real attention
  block in the loop" path.
* **M10 Long-Horizon Reconstruction V8** — 7-head LHR V8
  (causal + branch + cycle + merged-branch + cross-role +
  cross-cycle + **substrate-conditioned**) at ``max_k=48`` with
  degradation curve probe to ``k=96``. The substrate-conditioned
  head consumes the tiny-runtime hidden state as a conditioning
  input.
* **M11 ECC Codebook V8** — 7-level (K1=32 × K2=16 × K3=8 ×
  K4=4 × K5=2 × K6=2 × K7=2 = 131072 codes ≈ 17 data bits per
  septuple) + BCH(31,16) per segment. Target ≥ 19 bits/visible-
  token at full emit.
* **M12 Transcript-vs-Shared-vs-Substrate Arbiter V5** — 6-arm:
  {transcript, shared, merge_consensus, trust_weighted_merge,
  **substrate_replay**, abstain}. ``substrate_replay`` injects
  the latent into the tiny KV bank and lets the substrate
  forward generate the answer; this is the first capsule-vs-
  substrate head-to-head in the programme.

W56 is the first **executable substrate-coupling** milestone in
the programme; it is NOT a claim of third-party transformer-
internal access. ``W56-L-NO-THIRD-PARTY-SUBSTRATE-COUPLING-CAP``
names this honestly. The W56-C-DEEP-TRANSFORMER-COUPLING
conjecture (the third-party hosted-model substrate coupling) is
**carried forward**; W56 raises the in-repo substrate floor but
does not close the third-party side.

## H-bars (42 total)

H-bars enumerate the empirical content of W56. Pre-committed; if
3-seed mean falls below the bar, the corresponding limitation
theorem fires.

| H-bar | Bar | Family / family-row |
|-------|-----|---------------------|
| H1  | Trivial W56 envelope is byte-identical to W55 outer when all M flags are disabled | R-113 `trivial_w56_passthrough` |
| H2  | Tiny substrate forward determinism: two forward passes on identical token sequences produce byte-identical hidden states and logits | R-113 `tiny_substrate_forward_determinism` |
| H3  | Tiny substrate KV-cache reuse: incremental decode over a sequence yields identical logits to a from-scratch forward | R-113 `tiny_substrate_kv_cache_reuse` |
| H4  | Tiny substrate causal mask soundness: position `i` does not attend to position `j>i` (attention weights at upper-triangle = 0) | R-113 `tiny_substrate_causal_mask_soundness` |
| H5  | Substrate adapter capability matrix correctly classifies (tiny_runtime, synthetic, ollama-like) into {substrate_full, text_only, logits_only} | R-113 `substrate_adapter_capability_matrix` |
| H6  | KV bridge inject + forward changes logits: injected capsule carrier strictly perturbs at least one logit beyond `1e-6` from the un-injected baseline | R-113 `kv_bridge_injection_perturbs_logits` |
| H7  | KV bridge replay determinism: same capsule + same prompt + same RNG → byte-identical logits | R-113 `kv_bridge_replay_determinism` |
| H8  | Persistent V8 4-skip 64-turn cosine gain ≥ 0.3 vs V7 triple-skip on a long-horizon distractor regime | R-113 `persistent_v8_quad_skip_gain` |
| H9  | Multi-hop V6 8-backend chain-length-7 fidelity ≥ 0.45 (lower than V5 due to capacity; honestly cap) | R-113 `multi_hop_v6_oct_chain_len7_transitivity` |
| H10 | MLSC V4 substrate-witness binds to substrate hidden state hash and walks merge ancestors | R-113 `mlsc_v4_substrate_witness_round_trip` |
| H11 | Consensus controller V2 6-stage chain completes (every stage attempted) and audit trail records each stage | R-113 `consensus_controller_v2_6stage_audit` |
| H12 | Disagreement algebra V2 enriched identity checks (idempotent ⊕ + self-cancellation + distributivity-on-agreement + substrate-projection) pass on probe inputs | R-113 `disagreement_algebra_v2_identities` |
| H13 | Deep substrate hybrid forward reads real KV: ablating the substrate attention block changes the output relative to capsule-only baseline | R-114 `deep_substrate_hybrid_kv_read` |
| H14 | Deep substrate hybrid forward writes real KV: a write at turn `t` shows up in the KV cache at turn `t+1` | R-114 `deep_substrate_hybrid_kv_write` |
| H15 | Persistent V8 96-turn finite-recall soundness 1.0 (recall ≥ 0 on every regime) | R-114 `persistent_v8_96turn_finite_recall` |
| H16 | Persistent V8 128-turn stretch soundness 1.0 | R-114 `persistent_v8_128turn_stretch` |
| H17 | LHR V8 substrate-conditioned head MSE ≤ 0.70 at k=36 | R-114 `lhr_v8_substrate_conditioned_recovers_t_minus_36` |
| H18 | LHR V8 max_k=48 head MSE ≤ 1.50 at k=48 | R-114 `lhr_v8_k48_stretch` |
| H19 | LHR V8 degradation curve: min MSE in `k ∈ {1..96}` ≤ 1.0 | R-114 `lhr_v8_degradation_curve` |
| H20 | ECC V8 bits/visible-token ≥ 19.0 at full emit (3-seed mean) | R-114 `ecc_v8_compression_19_bits` |
| H21 | ECC V8 rate-floor falsifier: target 128 bits/token is missed by every seed (cap reproduces) | R-114 `ecc_v8_rate_floor_falsifier` |
| H22 | TVS arbiter V5 6-arm pick rates sum to 1.0; substrate_replay arm is non-empty in at least one cell | R-114 `tvs_arbiter_v5_6arm_dominance` |
| H23 | Uncertainty layer V4 substrate-conditioned composite confidence reproduces W55 composite when substrate signal is zero | R-114 `uncertainty_v4_substrate_zero_passthrough` |
| H24 | BCH(31,16) triple-bit correct rate ≥ 0.80 (3-seed mean; honest cap) | R-115 `bch_31_16_triple_bit_correct` |
| H25 | BCH(31,16) four-bit detect rate ≥ 0.50 (honest bar; some 4-bit patterns can mis-correct) | R-115 `bch_31_16_four_bit_detect` |
| H26 | CRC V4 silent failure rate ≤ 0.02 (tighter than W55's 0.03) | R-115 `crc_v4_silent_failure_floor` |
| H27 | CRC V4 5-bit burst recovery via 2D interleaving ≥ 0.90 | R-115 `crc_v4_2d_interleave_burst_recovery` |
| H28 | Consensus V2 substrate-conditioned tiebreaker recall ≥ 0.70 on a split-vote regime | R-115 `consensus_v2_substrate_tiebreaker_recall` |
| H29 | MLSC V4 trust × algebra-signature decay correct (decay-then-reinforce arithmetic) | R-115 `mlsc_v4_trust_algebra_decay` |
| H30 | Disagreement algebra V2 substrate-projection identity: `(a ⊕ b) projected_to_substrate ≈ substrate_forward(a ⊕ b)` cosine ≥ 0.5 on a probe distribution | R-115 `disagreement_algebra_v2_substrate_projection` |
| H31 | Compromise V8 persistent state forge protect rate ≥ 0.55 (honest bar; weaker than V7 due to additional substrate adversary axis) | R-115 `compromise_v8_persistent_state` |
| H32 | CRC V4 safety: silent_failure rate ≤ 0.02 under stress 5-bit corruption | R-115 `corruption_robust_carrier_v4_safety` |
| H33 | Uncertainty V4 substrate-weighted composite penalises low-substrate-fidelity components | R-115 `uncertainty_v4_substrate_weighted_penalises` |
| H34 | Persistent V8 chain walk depth ≥ 64 turns under a 72-turn run | R-115 `persistent_v8_chain_walk_depth` |
| H35 | W56 integration envelope verifier accepts a complete W56 turn bundle and rejects each of the 38 enumerated forgery modes | R-115 `w56_integration_envelope` |
| H36 | TVS arbiter V5 budget allocator distributes total_budget across all 6 arms with sum-to-1 fractions | R-115 `arbiter_v5_budget_allocator` |
| H37 | Deep substrate hybrid adaptive abstain threshold strictly monotone in input L2 norm (carries forward W55 with substrate signal) | R-115 `deep_substrate_hybrid_adaptive_abstain` |
| H38 | CRC V4 2D interleaving cell-correctness: every (row, col) corruption pattern of weight ≤ 1 per row + ≤ 1 per col is recoverable | R-115 `crc_v4_2d_interleave_cell_correctness` |
| H39 | MLSC V4 per-fact provenance_chain walks back to root capsule | R-115 `mlsc_v4_per_fact_provenance_walk` |
| H40 | Substrate-conditioned reconstruction strictly improves over proxy-only LHR V8 on a long-horizon regime where the substrate has seen the carrier | R-115 `lhr_v8_substrate_vs_proxy_only_recovery` |
| H41 | Substrate KV bank writes survive across `n_turns ≥ 8` (real cross-turn KV reuse, not transcript replay) | R-115 `substrate_kv_cross_turn_reuse` |
| H42 | Transcript-vs-shared-state-vs-substrate three-way: substrate_replay arm is preferred over transcript in at least one fixed-budget cell | R-115 `tvs_arbiter_v5_substrate_preferred_over_transcript` |

## Strong / partial / failure verdicts

* **Strong success**: ≥ 36 / 42 H-bars met across 3 seeds.
  Materially advances Context Zero by adding a real, executable
  substrate floor. The first milestone that can honestly say
  "the KV cache is real here".
* **Partial success**: 24 / 42 ≤ H-bars met < 36.
  Substrate runtime is real but the bridge or one of the
  V8 stacks does not lift end-to-end behaviour; honest scope
  reduction.
* **Failure**: < 24 / 42 H-bars met.
  Substrate runtime did not provide enough surface to lift the
  capsule layer; W56-L falsifier fires and W56 is recorded as
  capsule-only refinement, not substrate-coupling.

## Honest scope (forbidden phrasings)

* *"W56 is real transformer-internal coupling on Ollama / OpenAI / any
  hosted model"* — **forbidden**. The substrate is a tiny in-repo
  runtime. Third-party hosted-model substrate access remains
  substrate-blocked.
* *"W56 closes W43-C-MIXED-CURVATURE-LATENT / W43-C-COLLECTIVE-KV-POOLING
  / W47-C-DEEP-TRANSFORMER-COUPLING / ... / W55-C-DEEP-TRANSFORMER-COUPLING"* —
  **forbidden**. These remain substrate-blocked on third-party
  models. W56 further-bounds them by raising the in-repo
  substrate floor but does not close them.
* *"W56 is GPU-accelerated"* — **forbidden**. Tiny substrate is
  CPU NumPy. ``W56-L-NUMPY-CPU-TINY-SUBSTRATE-CAP`` names this.
* *"W56 is a frontier model"* — **forbidden**. Default tiny
  substrate is 2-layer, 4-head, d_model=32, vocab=259 (byte
  vocab + a few control tokens). Honest small-research-runtime
  scope.
* *"W56 makes real LLMs condition on the KV bridge"* —
  **forbidden**. Only the tiny in-repo substrate is bridged.
  ``W56-L-REAL-LLM-KV-BRIDGE-CAP`` documents that hosted
  backends cannot accept KV injection through their HTTP
  surface.
* *"W56 bumps the version"* — **forbidden**.
  ``coordpy.__version__`` remains ``0.5.20``; no PyPI release;
  W56 ships at explicit-import paths only.

## Mechanical determinism

* All W56 modules use the W47 pure-Python autograd ``Variable``
  + ``AdamOptimizer`` (or pure-NumPy for the tiny substrate),
  with deterministic seeded init.
* Tiny substrate is fully deterministic given a seed.
* Replay determinism: identical inputs + seed → byte-identical
  envelope bytes.
* KV bridge inject + forward: deterministic given inject payload
  + prompt + seed.

## Cumulative trust boundary

Adds 38 new disjoint envelope failure modes; cumulative trust
boundary across W22..W56 = **524 enumerated failure modes** (486
from W22..W55 + 38 new at W56).

## Repository hygiene

* No version bump.
* No PyPI release.
* No Claude / AI authorship trailers.
* W56 modules at explicit-import paths only; not re-exported
  through ``coordpy.__init__`` or ``coordpy.__experimental__``.
* Released v0.5.20 wheel surface byte-for-byte unchanged.
