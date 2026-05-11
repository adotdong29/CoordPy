# Pre-committed success criterion — W49 Multi-Block Cross-Bank Coordination (MBCC)

> Programme step: post-W48. Mints axis 46 of the Context Zero
> programme. Strictly additive on top of W48 SSTP, W47 AMS, W46
> MMC, W45 LMC, W44 LMCC, W43 PMC, and the released v3.43 line.
> Honest scope: W49 stacks **multiple** trainable transformer
> blocks (not one), runs **role-conditioned multi-bank pseudo-KV**
> (not a single bank), trains a **learned eviction policy** (not
> FIFO), trains a **retention head** (a separate trainable
> question-answering head against the bank), trains a **dictionary
> codebook** that compresses the latent-control bytes further, and
> evolves a **content-addressed shared-latent capsule per turn**
> (an explicit latent surrogate, distinct from W48's per-role
> rank-`r` delta). It does NOT touch transformer-internal hidden
> state, real KV cache bytes, attention weights, or embeddings.
> W43..W48's substrate-blocked conjectures
> (`W43-C-MIXED-CURVATURE-LATENT`,
> `W43-C-COLLECTIVE-KV-POOLING`,
> `W43-C-FULL-GRASSMANNIAN-HOMOTOPY`,
> `W47-C-DEEP-TRANSFORMER-COUPLING`,
> `W48-C-REAL-KV-COUPLED-PROXY`) carry forward unchanged. W49 is
> the strongest *executable proxy* line we can write today; it is
> *not* a closure of those.

## Mechanism

W49 introduces the **Multi-Block Cross-Bank Coordination (MBCC)**
layer — the first capsule-native CoordPy layer where:

1. **A stacked, residual, multi-block proxy transformer.** `L_p`
   stacked blocks (default 2). Each block is (i) a W48-style
   multi-head proxy attention sub-layer followed by (ii) a
   trainable position-wise tanh feed-forward sub-layer, both
   wired with residual connections and trainable LayerNorm-style
   scaling. Each block has its own trainable weights. The block
   stack consumes the W48 query and feeds the W48 reconstruction
   + gate logit.

2. **Multi-bank role-conditioned pseudo-KV.** Per-role banks
   (one bank per ``role_universe`` element) plus a single
   **shared** team-wide bank. Reads aggregate over the role
   bank + the shared bank with a learned per-read **bank-mix
   gate**; writes are routed by a learned **bank-router**
   sigmoid (writes either to the role bank, the shared bank, or
   both). Strict causal mask per bank.

3. **Learned eviction policy.** A trainable per-slot **eviction
   score** (small sigmoid head fed `(slot_age, role_match,
   write_gate)`) that decides which slot to evict when a bank is
   at capacity — replaces W48's plain ring-buffer FIFO.

4. **Retention/recall head.** A separate trainable two-layer
   head that, given the **current** shared state + flat
   channels + multi-bank read, predicts whether a hash-keyed
   **target fact** was previously written into a bank. Trained
   on a synthetic retention regime that pairs each input with a
   binary "should the proxy recall this?" label.

5. **Dictionary codebook compression.** A trainable
   `K_dict`-prototype codebook over the latent-control output;
   the latent payload is quantised to the nearest codebook entry
   and emitted as a single token referring to the codebook
   index. **Strictly stronger** compression than W48 (which
   emitted one bit per gate); a packed multi-token
   `LATENT_CTRL_V2` block carries `code` + `mask` + `bits`.

6. **Content-addressed shared-latent capsule per turn.** A new
   capsule (`SharedLatentCapsule`) whose value at turn `t` is
   the trained projection of the multi-block output of turn
   `t-1`; CID is content-addressed; chain-walk recovers all
   prior latent states. **Distinct** from W48's static shared
   base state vector — the W49 latent state *evolves
   deterministically across turns* and is bound under the
   envelope chain.

7. **Cramming witness.** A per-turn `CrammingWitness` records
   (i) the bits packed into the dictionary code + emit mask +
   bits payload, (ii) the visible-token cost of the W49 control
   block, (iii) the shared-latent capsule size in bytes, and
   (iv) the implied **structured-bits / visible-token** ratio.

8. **Training trace witness.** Identical envelope role to W47's
   and W48's. Carries forward seed + n_steps + optimiser config +
   loss + grad-norm + final params CID. The joint loss is the
   sum of: classifier BCE, reconstruction L2, retention BCE,
   dictionary cross-entropy, eviction BCE.

9. **`MultiBlockProxyTeam` orchestrator.** Sits beside W48
   `SharedStateProxyTeam`. Reduces to it byte-for-byte under a
   trivial config (the `W49-L-TRIVIAL-MULTI-BLOCK-PASSTHROUGH`
   falsifier).

All of it is pure-Python / stdlib. No NumPy, no PyTorch, no JAX
dependency. The pure-Python reverse-mode autograd engine from W47
(`Variable` + `AdamOptimizer`) is reused unchanged. The released
SDK v3.43 contract remains byte-for-byte unchanged.

Honest scope (do-not-overstate)
-------------------------------

W49 does NOT touch transformer-internal hidden state, KV cache
bytes, attention weights, or embeddings. Every parameter of the
multi-block proxy operates over W43 capsule-layer encodings, the
W47 trainable channel features, and the (now multi-bank) pseudo-
KV factor banks' *capsule-layer* slots. The pseudo-KV banks
reproduce the algebraic interface of a per-role + shared KV cache
at the capsule layer; they do **not** transplant real KV state
from a transformer's attention layers. The substrate-blocked
W43..W48 conjectures are unchanged.

W49 does NOT claim full-precision parity with a real
transformer block stack. The multi-block proxy stack uses tanh
nonlinearities, a capsule-layer-only attention pool over the
banks, and trains on synthetic regimes. It is the strongest
honest *executable proxy* we can write at the capsule layer
under W47's pure-Python autograd cap.

W49 does NOT claim CUDA / GPU acceleration. The pure-Python
autograd engine reused from W47 is correct but slow; W49 trains
on bounded synthetic banks (~16–32 examples for ~40–120 Adam
steps).

W49 does NOT claim adversarial robustness under
training-distribution forgery —
`W49-L-MULTI-BLOCK-DISTRIBUTION-CAP` (strengthens
`W48-L-PROXY-DISTRIBUTION-CAP`).

W49 is strictly additive. When configured trivially, the
`MultiBlockProxyTeam` orchestrator reduces to
`SharedStateProxyTeam.run` byte-for-byte — the
`W49-L-TRIVIAL-MULTI-BLOCK-PASSTHROUGH` falsifier.

This module lives at `coordpy.multi_block_proxy` and is NOT
exported through `coordpy.__experimental__` at this milestone;
the stable v0.5.20 SDK contract is preserved byte-for-byte.
Sophisticated callers reach the W49 surface through an
explicit `from coordpy.multi_block_proxy import ...` import.

## H1..H16 success bar

Sixteen pre-committed hypotheses across **two benchmark families**
(R-96 + R-97); each is exercised by a per-family test in
`tests/test_r96_benchmark.py` / `tests/test_r97_benchmark.py`
plus per-component unit coverage in
`tests/test_multi_block_proxy_w49.py`.

### H1 — Trivial multi-block passthrough (R-96)

A trivially-configured `MultiBlockProxyRegistry`
(`multi_block_enabled=False`, `multi_bank_enabled=False`,
`retention_enabled=False`, `dictionary_enabled=False`,
`shared_latent_capsule_enabled=False`, W48-trivial inner) reduces
to `SharedStateProxyTeam.run` byte-for-byte. The
`r96_trivial_multi_block_passthrough` family reports
`passthrough_ok = 1.0` across all eight W43..W48 arms and the
new `w49_multi_block` arm across all three seeds.

### H2 — Multi-block depth strictly beats single block (R-96)

On a synthetic two-step composition regime where the target gate
depends on a *composed* nonlinear function of two channels (the
W47 / W48 single-block stack cannot fit it cleanly), an
`L_p = 2` multi-block stack achieves classifier accuracy > the
`L_p = 1` (W48) baseline on a per-seed margin ≥ 0.10. The
`r96_multi_block_depth` family;
`acc(L_p=2) - acc(L_p=1) ≥ 0.10` on the mean across seeds.

### H3 — Multi-bank role-conditioned pseudo-KV beats single bank (R-96)

On a regime where each role *owns its own* fact (role-A writes
fact alpha, role-B writes fact beta), the per-role bank read
recovers the role's own fact with mean cosine similarity ≥ 0.80
when the W48 single-bank baseline at the same capacity
saturates and confuses facts (cosine ≤ 0.50). The
`r96_multi_bank_recall` family;
`multi_bank_recall - w48_single_bank_recall ≥ 0.20` on mean
across seeds.

### H4 — Learned eviction beats FIFO (R-96)

On a regime where the *first* slot carries a signal fact and the
*last few* slots carry noise, the learned eviction policy keeps
the signal slot under a tight bank capacity where W48's plain
FIFO drops it. The `r96_learned_eviction` family;
`recall_after_overflow(learned) - recall_after_overflow(fifo)
≥ 0.20` on mean across seeds.

### H5 — Retention head answers correctly (R-96)

The trained retention head answers the binary "was this fact
stored?" question with accuracy ≥ 0.85 across a synthetic
mixed-fact regime. The `r96_retention_head` family;
`retention_acc ≥ 0.85` AND `retention_acc - chance ≥ 0.30`.

### H6 — Dictionary codebook compression saves bits (R-96)

The trained codebook over the latent-control payload reduces the
mean visible-token cost of the model-facing control block by
≥ 25% vs the W48 packed `LATENT_CTRL` block, while still
round-tripping bijectively through the witness. The
`r96_dictionary_compression` family;
`(w48_ctrl_tokens - w49_ctrl_tokens) / w48_ctrl_tokens ≥ 0.25`
AND `dictionary_round_trip_ok = 1.0`.

### H7 — Shared-latent capsule evolves and is recoverable (R-96)

The W49 shared-latent capsule's CID at turn `t` differs from
turn `t-1` (the latent evolves) and is recoverable from the
envelope chain alone (chain-walk from turn `t` recovers the
turn-`t-1` capsule CID byte-for-byte). The
`r96_shared_latent_capsule` family;
`latent_evolves_ok = 1.0` AND `chain_walk_recovers_ok = 1.0`.

### H8 — Cross-bank causal interference is bounded (R-96)

On a regime where a forged write into role-A's bank at turn `t`
must not corrupt role-B's read at turn `t+1`, the W49
multi-bank arm's role-B read shows L2 perturbation ≤ 0.05
relative to the no-forgery baseline. The
`r96_cross_bank_interference` family;
`role_b_perturbation ≤ 0.05` (proves that bank routing isolates
roles).

### H9 — Replay determinism across the multi-block stack (R-96)

Two independent runs of `MultiBlockProxyTeam.run` with the same
training set, seed, registry, and observation builder produce
byte-identical `final_output`, root CID, every
`multi_block_outer_cid`, every per-role
`pseudo_kv_bank_head_cid`, the shared-latent capsule CID list,
and the dictionary codebook CID. The `r96_replay_determinism`
family; `replay_determinism_ok = 1.0` across all seeds.

### H10 — W49 envelope verifier soundness (R-96)

The W49 verifier rejects 18+ disjoint forged envelopes (schema
mismatch, multi-block params CID mismatch, role-bank CID
mismatch, shared-bank CID mismatch, shared-latent capsule CID
mismatch, dictionary CID mismatch, retention head CID mismatch,
eviction policy CID mismatch, outer-CID mismatch, etc.).
Cumulative trust boundary across W22..W49 = **319 enumerated
failure modes** (301 from W22..W48 + 18 new at W49).

### H11 — Long-branch retention (R-97)

On a length-12 branch path with a target fact emitted at turn 0,
the W49 retention head + multi-bank read jointly recover the
fact at turn 11 with accuracy ≥ 0.80; the W48 baseline drops
below 0.40 on the same regime because W48's single bank
overflows. The `r97_long_branch_retention` family;
`w49_acc ≥ 0.80` AND `w49_acc - w48_acc ≥ 0.30`.

### H12 — Cycle/consensus reconstruction (R-97)

On a regime where the same `(branch_id, cycle_id)` recurs every
3 turns, the W49 reconstruction decoder + multi-block stack
recovers the cycle's first-emission feature vector with
L1 ≤ 0.30 at every recurrence; the W48 single-block decoder
stays above 0.50. The `r97_cycle_reconstruction` family;
`w49_recon_l1 ≤ 0.30` AND `w49_recon_l1 ≤ 0.7 * w48_recon_l1`.

### H13 — Cramming structured-bits ratio (R-97)

W49 achieves a strictly higher **structured-bits-per-visible-
token** ratio than W48 on a fixed-length task (the W49 control
carries dictionary code + mask + bits; W48 carries only emit
mask + bits). The `r97_cramming_bits_ratio` family;
`w49_bits_per_token > w48_bits_per_token` AND
`w49_bits_per_token ≥ 1.5 * w48_bits_per_token` on mean across
seeds.

### H14 — Shared-state-vs-transcript replay comparison (R-97)

On a regime where the team must answer a question that depends
on facts emitted at turn 0..3, the W49 shared-latent +
multi-bank arm hits task-correct rate ≥ 0.85 with the W42
transcript-replay path receiving only the bounded N=2 visible
handoffs (so the transcript path *cannot* see the original
facts). The `r97_shared_state_vs_transcript` family;
`w49_correct - transcript_correct ≥ 0.40`.

### H15 — Aggressive-compression partial recovery (R-97)

Under aggressive compression (W49 emits **only** the dictionary
code and a 1-bit mask), the reconstruction decoder still
recovers the prior-turn flat features at L1 ≤ 0.45 (W48 with
the same packed-control byte budget collapses to ≥ 0.80). The
`r97_aggressive_compression` family;
`w49_recon_l1 ≤ 0.45` AND `w49_recon_l1 ≤ 0.7 * w48_recon_l1`.

### H16 — Multi-block distribution cap reproduces (R-97)

Adversarial all-channel forgery + forged per-role banks + forged
training distribution + forged shared-latent capsule: the W49
multi-block stack cannot recover. The
`r97_multi_block_distribution_cap` family;
`downstream_protect_rate ≤ 0.3` across all seeds — proved-
conditional limitation `W49-L-MULTI-BLOCK-DISTRIBUTION-CAP`
(strengthens `W48-L-PROXY-DISTRIBUTION-CAP`).

## Falsifiers

* **W49-L-TRIVIAL-MULTI-BLOCK-PASSTHROUGH** — a trivially-
  configured `MultiBlockProxyRegistry` reduces to
  `SharedStateProxyTeam.run` byte-for-byte; if H1 fails, the
  trivial-passthrough property is falsified.

* **W49-L-MULTI-BLOCK-DISTRIBUTION-CAP** — adversarial all-
  channel forgery + forged role banks + forged training set:
  the trained multi-block stack cannot recover because it
  learned the adversary's distribution. Reproduces honestly in
  the R-97 family.

* **W49-L-NO-REAL-KV-CAP** — the W49 stack still does not touch
  transformer-internal KV bytes; it reproduces the algebraic
  interface at the capsule layer with auditable byte-deterministic
  provenance, no more. Multi-bank + multi-block does not change
  this.

* **W49-L-PURE-PYTHON-TRAINING-COST-CAP** — the pure-Python
  autograd engine carries forward from W47; multi-block training
  is approximately `L_p` × W48 cost per step. Caps practical
  n_steps at a few dozen on the per-family wall-clock budget.

## Per-component verdicts (preview)

* **Multi-block proxy transformer (`L_p ≥ 2`)** — *behaviourally
  useful* if H2 passes; *structurally useful* regardless (each
  block has its own CID).
* **Multi-bank role-conditioned pseudo-KV** — *behaviourally
  useful* (per-role recall ≥ 0.80, cross-bank interference
  ≤ 0.05).
* **Learned eviction policy** — *behaviourally useful* on H4;
  *structurally useful* (eviction-witness CID).
* **Retention head** — *behaviourally useful* (binary
  recall ≥ 0.85).
* **Dictionary codebook compression** — *behaviourally useful*
  on H6 (≥ 25% token savings); *structurally useful*
  (round-trips through codebook CID).
* **Shared-latent capsule per turn** — *behaviourally useful*
  if the latent state changes downstream behaviour;
  *structurally useful* always (auditable evolution).
* **Cramming witness** — *structurally useful* (auditable
  bits/token frontier).
* **Training trace witness** — *structurally useful* (full
  auditability of the training run, carries forward from W47).
* **Multi-block distribution cap** — *limitation reproduces
  honestly*.

## Architecture triage

| Frontier candidate                                  | W49 bucket                                              | Verdict |
|---|---|---|
| Multi-block stacked transformer proxy               | **transformer-proxy now (depth)**                       | shipped |
| Role-conditioned multi-bank pseudo-KV                | **transformer-proxy now (banks)**                       | shipped |
| Trainable slot eviction policy                      | **trainable now**                                       | shipped |
| Retention/recall head                                | **trainable now**                                       | shipped |
| Dictionary codebook compression                      | **trainable now (compression)**                         | shipped |
| Shared-latent capsule per turn                       | **trainable + content-addressed now**                   | shipped |
| Cramming witness (bits/token frontier)               | **structural now**                                      | shipped |
| Cross-bank causal interference theorem               | **proved-conditional**                                  | shipped |
| Real KV-cache pooling across turns                   | **substrate-blocked**                                   | unchanged |
| Transformer-internal mixed-curvature attention       | **substrate-blocked**                                   | unchanged |
| True hidden-state / KV sharing                       | **substrate-blocked**                                   | unchanged |
| Multi-host shared-state transfer                     | **substrate-blocked**                                   | unchanged |
| GPU/CUDA-backed autograd                             | **substrate-blocked (deliberately deferred)**           | unchanged |

## What W49 explicitly does NOT do

* W49 does NOT close `W47-C-DEEP-TRANSFORMER-COUPLING`,
  `W48-C-DEEP-TRANSFORMER-COUPLING`, or any other W43..W48
  substrate-blocked direction. The multi-block + multi-bank
  proxy is the strongest *capsule-layer proxy* we can write
  today; closing the conjectures still requires architectural
  access to the transformer's attention computation and KV
  cache.
* W49 does NOT transplant real KV-cache bytes. The multi-bank
  pseudo-KV factor banks are capsule-layer surrogates that
  reproduce the algebraic interface; they never read or write a
  real transformer KV cache.
* W49 does NOT claim multi-host coupling. The shared-latent
  capsule + multi-bank pseudo-KV banks run on a single process
  / single backend.
* W49 does NOT claim training-data-free generalisation. The
  multi-block proxy is trained on hermetic synthetic banks
  pre-committed in the R-96 / R-97 sources.
* W49 does NOT close `W47-C-LIVE-MULTI-HOST-AUTOGRAD`,
  `W47-C-GPU-BACKED-AUTOGRAD-SDK`,
  `W48-C-REAL-KV-COUPLED-PROXY`, or
  `W48-C-MULTI-HOST-SHARED-STATE`.
* W49 does NOT ship CUDA / GPU support.

## Version + release status

* **No version bump**: `coordpy.__version__` remains `"0.5.20"`.
* **No SDK bump**: `coordpy.SDK_VERSION` remains
  `"coordpy.sdk.v3.43"`.
* **No PyPI release**: no wheel built, no upload step, no
  release tag pushed.
* **No new public symbol** added to `coordpy/__init__.py`. The
  W49 module ships at `coordpy.multi_block_proxy` and is
  reachable only through an explicit import — same convention
  as W43..W48.

## New theorem-style claims (preview)

* **W49-T-MULTI-BLOCK-DEPTH-EXPRESSIVITY** (proved-conditional +
  empirical) — under the bounded-feature assumption + the
  two-step-composition regime, an `L_p = 2` multi-block proxy
  strictly separates a composition that a single block (W48)
  cannot.
* **W49-T-MULTI-BANK-CAUSAL-INTERFERENCE-BOUND** (proved by
  inspection + mechanically-checked) — for any pair of roles
  `(r_a, r_b)` with `r_a ≠ r_b` and admissible turns
  `t_a < t_b`, the W49 read at turn `t_b` from `r_b`'s bank
  cannot draw from a slot whose `role == r_a` unless the
  shared bank explicitly received that slot.
* **W49-T-LEARNED-EVICTION-SELECTIVITY** (proved-conditional +
  empirical) — the trained eviction policy retains higher-
  signal slots over noise slots under tight capacity at
  selectivity > 0.5 on the alternating signal/noise regime.
* **W49-T-RETENTION-HEAD-CORRECTNESS** (proved-conditional +
  empirical) — the trained retention head answers "was this
  fact stored?" with ≥ 0.85 accuracy on the synthetic regime.
* **W49-T-DICTIONARY-COMPRESSION-RATE** (proved + empirical)
  — the trained codebook reduces mean control-block token
  count by ≥ 25% vs W48's packed control while round-tripping
  bijectively through the witness CID.
* **W49-T-SHARED-LATENT-CHAIN-WALK** (proved by inspection +
  mechanically-checked) — given the W49 envelope chain alone,
  an auditor can recover every prior turn's shared-latent
  capsule CID by chain-walking; no external state required.
* **W49-T-CRAMMING-WITNESS-SOUNDNESS** (proved by inspection
  + mechanically-checked) — the per-turn cramming witness's
  `structured_bits` field equals the sum of the dictionary-
  code bits + emit-mask bits + bits-payload bits exactly.
* **W49-T-MULTI-BLOCK-TRAIN-DETERMINISM** (proved +
  mechanically-checked, carries forward from W47/W48) — two
  independent training runs produce byte-identical parameter
  CIDs and trace CIDs.
* **W49-T-VERIFIER-SOUNDNESS** (proved by inspection +
  mechanically-checked) — the W49 verifier enumerates 18
  disjoint failure modes; cumulative trust boundary across
  W22..W49 = 319 modes.
* **W49-T-LONG-HORIZON-RETENTION** (proved-conditional +
  empirical) — on a length-12 branch path the W49 retention
  arm beats W48's by ≥ 0.30 absolute accuracy.
* **W49-T-AGGRESSIVE-COMPRESSION-RECOVERY** (proved-conditional
  + empirical) — under aggressive compression (dictionary code
  + 1-bit mask only), the W49 reconstruction decoder still
  recovers prior-turn features at L1 ≤ 0.45.
* **W49-L-TRIVIAL-MULTI-BLOCK-PASSTHROUGH** (proved by
  inspection + empirical) — trivial W49 = W48 byte-for-byte.
* **W49-L-NO-REAL-KV-CAP** (carries forward, strengthens W48)
  — multi-bank does not transplant real KV bytes.
* **W49-L-MULTI-BLOCK-DISTRIBUTION-CAP** (proved-conditional
  limitation, strengthens W48) — under adversarial training-
  distribution forgery W49 cannot recover.
* **W49-L-PURE-PYTHON-TRAINING-COST-CAP** (carries forward) —
  multi-block training is roughly `L_p` × W48 cost per step.
* **W49-L-CTRL-AWARE-MODEL-INDIFFERENCE-CAP** (carries forward
  from W48) — real LLMs may or may not condition on
  `LATENT_CTRL_V2`. H14 evidence is anchored to the
  `MultiBlockAwareSyntheticBackend`.
* **W49-C-DEEP-TRANSFORMER-COUPLING** (carries forward,
  bounds W48-C further) — the full direction of "train a
  deep, transformer-coupled controller that consumes hidden
  states and emits attention-mask adjustments / real KV-cache
  routing" remains substrate-blocked.
* **W49-C-CROSS-MODEL-LATENT-TRANSFER** (new conjectural
  direction) — sharing the W49 shared-latent capsule chain
  across two backends with different tokenizers is structurally
  compatible with the envelope chain but requires a
  tokenizer-agnostic adapter on the backend side. Out of W49
  scope.

## What this enables for the programme

* **Strengthens** the W47/W48 carry-forward
  `W48-C-DEEP-TRANSFORMER-COUPLING` by adding a *multi-block
  proxy stack + role-conditioned multi-bank pseudo-KV + learned
  eviction + retention head + dictionary compression* — the
  closest executable capsule-layer reconstruction of a deep
  transformer block stack we can write today.
* **Strengthens** `W48-L-PROXY-DISTRIBUTION-CAP` to include
  multi-bank forgery (`W49-L-MULTI-BLOCK-DISTRIBUTION-CAP`).
* **Preserves** all of W43..W48's deterministic-audit
  properties — the W49 module is strictly additive.
* **Does not close** the substrate-blocked W43..W48
  conjectures. The honest summary: W49 is the strongest
  *executable proxy* available without substrate access.
* **Mints** new W49 conjectures:
  `W49-C-DEEP-TRANSFORMER-COUPLING` (carry-forward, bounds W48
  further), and
  `W49-C-CROSS-MODEL-LATENT-TRANSFER` (cross-tokenizer latent
  transfer — needs backend support).

## Done = the following commits land

1. `coordpy/multi_block_proxy.py` ≈ 2200 LoC, pure Python /
   stdlib only. Reuses the W47 `Variable` + `AdamOptimizer`
   autograd engine via explicit import; reuses the W48
   `SharedStateProxyTeam` envelope via explicit import.
2. `coordpy/r96_benchmark.py` ≈ 1100 LoC, dependency-free.
3. `coordpy/r97_benchmark.py` ≈ 800 LoC, dependency-free.
4. `tests/test_multi_block_proxy_w49.py` — ≥ 30 tests covering
   every component + the trivial-passthrough falsifier +
   shared-latent capsule chain walk + multi-bank causal-
   interference bound + dictionary round-trip + retention
   head + eviction policy + multi-block determinism +
   verifier soundness.
5. `tests/test_r96_benchmark.py` — ≥ 10 tests covering H1..H10.
6. `tests/test_r97_benchmark.py` — ≥ 6 tests covering H11..H16.
7. `docs/RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md` and this
   success-criterion file.
8. Updates to `docs/RESEARCH_STATUS.md`,
   `docs/THEOREM_REGISTRY.md`, `docs/START_HERE.md`,
   `docs/HOW_NOT_TO_OVERSTATE.md`,
   `docs/context_zero_master_plan.md`,
   `papers/context_as_objects.md`, and `CHANGELOG.md`.
