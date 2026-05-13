# Start Here

CoordPy is a Python-first SDK and CLI for building auditable AI
agent teams with structured, content-addressed context. If you want
the fastest path to understanding what this repo ships and how to
use it, read this page first.

## What CoordPy is

CoordPy gives you a stable runtime contract for AI agent teamwork:

* **Bounded-context capsules instead of token cramming.** Prompts,
  responses, handoffs, and reports are stored as structured,
  content-addressed objects with provenance and budget metadata.
* **A reproducible runtime.** One `RunSpec` in, one `RunReport` out,
  with a sealed capsule graph you can inspect and verify.
* **A team-coordination layer.** Agents exchange `TEAM_HANDOFF`,
  `ROLE_VIEW`, and `TEAM_DECISION` capsules instead of ad hoc text.
* **An audit surface.** `coordpy-capsule verify` can re-hash the
  report and its artifacts from disk.

This repo also includes the full experimental research ladder under
`coordpy.__experimental__`, but the released product
surface is the stable SDK and CLI.

## Who it is for

CoordPy is for:

* developers building AI agent teams or LLM workflows that need
  reproducible shared context
* teams that want an audit trail instead of opaque prompt glue
* researchers who want the released system plus the benchmark and
  theorem trail behind it

If you only want the product surface, stay on the stable SDK and CLI
below. If you want the full research programme, jump to the paper and
results links in [Where to read next](#where-to-read-next).

## Install

CoordPy is on PyPI as
[`coordpy-ai`](https://pypi.org/project/coordpy-ai/) and imports as
`coordpy`. Requires Python 3.10 or newer.

```bash
pip install coordpy-ai
# or, isolated:
pipx install coordpy-ai
```

The CLI commands installed by the package are:

```bash
coordpy-team    # run / replay / sweep / compare an AgentTeam preset
coordpy-capsule # view / verify / verify-view / audit a sealed chain
coordpy         # run a research profile end to end
coordpy-import  # audit a SWE-bench-Lite-style JSONL
coordpy-ci      # apply the CI pass/fail gate to a finished report
```

Only required dependency is NumPy. Optional extras are available for
heavier local setups: `coordpy[scientific]`, `coordpy[dl]`,
`coordpy[heavy]`, `coordpy[crypto]`, `coordpy[docker]`,
`coordpy[dev]`.

## Minimal quickstart

The recommended front door is the `coordpy-team` CLI driving a
curated preset against a configured backend:

```bash
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:14b
export COORDPY_OLLAMA_URL=http://localhost:11434

coordpy-team run \
    --preset quant_desk \
    --task examples/scenario_bullish.txt \
    --out-dir /tmp/desk-run

coordpy-capsule verify-view \
    --view /tmp/desk-run/team_capsule_view.json
```

Equivalent Python path:

```python
from coordpy import AgentTeam, agent

team = AgentTeam.from_env(
    [
        agent("planner", "Break the task into steps."),
        agent("researcher", "Gather the facts."),
        agent("writer", "Write the final answer."),
    ],
    model="gpt-4o-mini",   # or qwen2.5:0.5b with COORDPY_BACKEND=ollama
    backend_name="openai", # or "ollama"
)
result = team.run("Explain what CoordPy does.")
print(result.final_output)
```

The structured-research path through the original `RunSpec` /
`RunReport` contract is still available and stable:

```python
from coordpy import RunSpec, run

report = run(RunSpec(profile="local_smoke", out_dir="/tmp/coordpy-smoke"))
assert report["readiness"]["ready"]
print(report["summary_text"])
```

Common environment variables:

```bash
# Local Ollama
export COORDPY_BACKEND=ollama
export COORDPY_MODEL=qwen2.5:0.5b
export COORDPY_OLLAMA_URL=http://localhost:11434

# OpenAI-compatible provider
export COORDPY_BACKEND=openai
export COORDPY_MODEL=gpt-4o-mini
export COORDPY_API_KEY=...
# Optional for compatible non-default providers:
# export COORDPY_API_BASE_URL=https://your-provider.example/v1
```

## Stable vs experimental

**Stable and released in SDK v3.43**

* `coordpy` SDK surface: `RunSpec`, `run`, `RunReport`,
  `SweepSpec`, `run_sweep`, `CoordPyConfig`, `Agent`, `AgentTurn`,
  `ActionDecision`, `AgentTeam`, `TeamResult`, `agent`,
  `create_team`, `replay_team_result`, `presets`,
  `TEAM_RESULT_SCHEMA`, `profiles`, `ci_gate`, `import_data`,
  `extensions`, capsule primitives, schema constants,
  `OpenAICompatibleBackend`, `backend_from_env`
* CLI surface: `coordpy-team`, `coordpy-capsule`, `coordpy`,
  `coordpy-import`, `coordpy-ci`
* On-disk schemas: `coordpy.capsule_view.v1`,
  `coordpy.team_result.v1`, `coordpy.provenance.v1`,
  `phase45.product_report.v2`

**Experimental but included**

* `coordpy.__experimental__`
* W22..W42 trust/adjudication and multi-agent coordination ladder
* R-69..R-89 benchmark drivers
* bounded live cross-host probes

**Out of scope for this release**

* `W42-C-NATIVE-LATENT`: transformer-internal trust transfer
* `W42-C-MULTI-HOST`: K+1-host disjoint topology beyond the current
  two-host setup

Those are next-programme architecture questions, not blockers to the
released CoordPy v3.43 line.

**Post-release research milestone — W43 Product-Manifold Capsule**

After 0.5.20 shipped, the next research step (W43) introduced the
**Product-Manifold Capsule (PMC)** layer: a six-channel mixed-
curvature decomposition (hyperbolic / spherical / euclidean
geometry + factoradic discrete control + subspace state +
causal lattice). W43 is held outside the stable SDK contract — it
ships at `coordpy.product_manifold` and is reachable only via
explicit import; the released v0.5.20 wheel's public surface is
byte-for-byte unchanged. See
[`RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md`](RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md)
for the full result note.

**Post-W43 research milestone — W44 Live Manifold-Coupled Coordination**

The next research step after W43 (W44) introduces the **Live
Manifold-Coupled Coordination (LMCC)** layer: the first
capsule-native CoordPy layer that lets three of the W43 channels
(spherical / subspace / causal) actually *change run behaviour*
in a sequential agent team via a manifold-conditioned gating
policy that substitutes a deterministic abstain output for the
agent's `generate()` call when the registered policy is violated.
A fourth channel (factoradic route) becomes a *live compressor*
that replaces the textual rendering of the role-arrival ordering
with a single integer header. The remaining two channels
(hyperbolic, euclidean) remain audit-only at the live layer. W44
is held outside the stable SDK contract — it ships at
`coordpy.live_manifold` and is reachable only via explicit import;
the released v0.5.20 wheel's public surface is byte-for-byte
unchanged. See
[`RESULTS_COORDPY_W44_LIVE_MANIFOLD.md`](RESULTS_COORDPY_W44_LIVE_MANIFOLD.md)
for the full result note.

**Post-W44 research milestone — W45 Learned Manifold Controller**

The next research step after W44 (W45) introduces the **Learned
Manifold Controller (LMC)** layer: the first capsule-native
CoordPy layer where the gating decisions themselves are *shaped
by data*. Five learned components — all closed-form-fittable in
pure NumPy-free Python: (i) learned channel encoder mapping each
of the six W43 channels to a fixed-dim feature vector (now
including the previously audit-only hyperbolic and euclidean
channels); (ii) attention-style softmax routing over channels;
(iii) shared-base + LoRA-style rank-1 role-specific adapter; (iv)
margin-calibrated gate via sigmoid; (v) a content-addressed
`MANIFOLD_HINT: route=<int> conf=<bucket> p=<prob>` prompt
control that real (or synthetic) LLM backends can read. W45 is
held outside the stable SDK contract — it ships at
`coordpy.learned_manifold` and is reachable only via explicit
import; the released v0.5.20 wheel's public surface is
byte-for-byte unchanged. See
[`RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md`](RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md)
for the full result note.

**Post-W47 research milestone — W48 Shared-State Transformer-Proxy**

The next research step after W47 (W48) introduces the
**Shared-State Transformer-Proxy (SSTP)** layer: the first
capsule-native CoordPy layer where a single team-shared base
state vector lives across turns and roles, a trainable
**pseudo-KV factor bank** of low-rank `(K, V)` tuples
reproduces the algebraic interface of a transformer KV cache
at the capsule layer (`softmax(Q·K^T/sqrt(d))·V` with strict
causal masking), an `H`-head **multi-head proxy attention
block** reads and writes the bank, a **slot-memory write head**
decides per turn whether to append a new slot, a
**reconstruction decoder** recovers prior-turn flat channel
features from the current shared state + pseudo-KV read, a
**branch/cycle-aware bias matrix** separates branches with
identical channel features, a **bijective branch-history
compressor** packs the team's branch path into a single
integer header with explicit visible-token savings, and a
**learned latent-control serializer** emits a single
`LATENT_CTRL: SHARED_STATE_HASH=... mask=... bits=...` line.
Eleven trainable, content-addressed components — all in pure
Python with no NumPy/JAX/PyTorch dependency, reusing the W47
`Variable` autograd engine + `AdamOptimizer`. W48 is the
strongest *executable proxy* for transformer-internal coupling
we can write today at the capsule layer; it does NOT touch
real KV bytes, hidden states, or attention weights. W48 is
held outside the stable SDK contract — it ships at
`coordpy.shared_state_proxy` and is reachable only via
explicit import; the released v0.5.20 wheel's public surface
is byte-for-byte unchanged. See
[`RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md`](RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md)
for the full result note.

**Post-W54 research milestone — W55 Deep Trust-Weighted Disagreement-Algebraic Latent Operating System**

The next research step after W54 (W55) introduces the **Deep
Trust-Weighted Disagreement-Algebraic Latent Operating System
(DTDA-LOS)** layer: eleven orthogonal capsule-native advances
on top of W54 Deep Mergeable Disagreement-aware LOS — (M1) a
5-layer **persistent latent state V7** with a *triple*
persistent skip-link (turn-0 anchor + fast EMA + slow EMA),
chain walks past **128 turns**, and a disagreement-algebraic
merge head; (M2) a **7-backend (A,B,C,D,E,F,G) hept multi-hop
translator V5** over 42 directed edges with chain-length-6
transitivity and **trust-weighted compromise arbitration**;
(M3) the **Mergeable Latent State Capsule V3 (MLSC V3)** —
extends MLSC V2 with **disagreement algebra primitives**
⊕/⊖/⊗, **per-fact confirmation count**, and **trust signature
decay**; (M4) a **Trust-Weighted Consensus Controller (TWCC)**
with continuous trust-weighted quorum and **5-stage decision
chain** {K-of-N → trust-weighted → best-parent → transcript →
abstain}; (M5) a **Corruption-Robust Carrier V3** — BCH(15,7)
**double-bit correction** + **5-of-7 majority repetition** +
**bit-interleaving**; (M6) a **depth-14 Deep Proxy Stack V6**
with **trust-projected residual gating**, **disagreement-
algebra head**, and **adaptive abstain threshold**; (M7) a
**6-head Long-Horizon Reconstruction V7** (causal + branch +
cycle + merged-branch + cross-role + cross-cycle) at
``max_k=36``; (M8) a **six-level ECC Codebook V7**
(K1=32 × K2=16 × K3=8 × K4=4 × K5=2 × K6=2 = 65536 codes)
plus BCH(15,7) per-segment; **18.333 bits/visible-token**;
(M9) a **5-arm Transcript-vs-Shared Arbiter V4** with per-arm
budget allocator; (M10) an **Uncertainty Layer V3** with
per-fact uncertainty propagation, **adversarial calibration
check**, and **trust-weighted composite**; (M11) a
first-class **Disagreement Algebra** module exposing ⊕/⊖/⊗
as content-addressed primitives with algebraic identities by
inspection. 33 new disjoint envelope failure modes
(cumulative trust boundary = **486 modes across W22..W55**).
W55 is held outside the stable SDK contract — it ships at
``coordpy.persistent_latent_v7``,
``coordpy.multi_hop_translator_v5``,
``coordpy.mergeable_latent_capsule_v3``,
``coordpy.trust_weighted_consensus_controller``,
``coordpy.corruption_robust_carrier_v3``,
``coordpy.deep_proxy_stack_v6``,
``coordpy.long_horizon_retention_v7``,
``coordpy.ecc_codebook_v7``,
``coordpy.transcript_vs_shared_arbiter_v4``,
``coordpy.uncertainty_layer_v3``,
``coordpy.disagreement_algebra``, and the composition
module at ``coordpy.w55_team``, reachable only via explicit
import; the released v0.5.20 wheel's public surface is
byte-for-byte unchanged. See
[`RESULTS_W55_DTDA_LOS.md`](RESULTS_W55_DTDA_LOS.md) for the
full result note.

**Post-W52 research milestone — W53 Persistent Mergeable Corruption-Robust Latent Operating System**

The next research step after W52 (W53) introduces the
**Persistent Mergeable Corruption-Robust Latent Operating
System (PMCRLOS)** layer: ten orthogonal capsule-native
advances on top of W52's eight. (i) A 3-layer **persistent
latent state V5** with a *persistent* skip-link applied at
every step plus a state-merge head. (ii) A **5-backend
multi-hop translator V3** over 20 directed edges with
chain-length-4 transitivity scoring and uncertainty-aware
arbitration that returns per-dim 1-sigma confidence
intervals. (iii) The **Mergeable Latent State Capsule
(MLSC)** load-bearing new abstraction: content-addressed
mergeable capsules with explicit ``MergeOperator`` and
content-addressed ``MergeAuditTrail``; supports K-of-N
consensus quorum with abstain. (iv) A depth-10 **deep proxy
stack V4** with merge-aware + corruption-aware heads.
(v) A **four-level ECC codebook V5** (K1=32 × K2=16 × K3=8
× K4=4 = 16384 codes) plus XOR parity bits per segment;
≥ 14.5 bits/visible-token at full emit (empirically 17.67)
plus single-bit corruption detection. (vi) A **four-headed
long-horizon reconstruction V5** (causal + branch + cycle +
merged-branch) at ``max_k=16`` with a degradation curve
probe across ``k ∈ {1..32}``. (vii) A **branch merge memory
V3** with consensus pages + content-addressed audit +
abstain semantics. (viii) A **corruption-robust carrier**
that composes ECC parity + 3-of-3 majority repetition over
the bits payload; reports detect / partial-correct /
abstain / silent-failure rates. (ix) A **transcript-vs-shared
arbiter V2** with explicit per-turn policy over {transcript,
shared, abstain} + oracle-correctness comparison. (x) An
**uncertainty / confidence layer** that composes
per-component confidences into a composite scalar +
calibration check. 30 new disjoint envelope failure modes
(cumulative trust boundary = **423 modes across W22..W53**).
W53 is held outside the stable SDK contract — it ships at
`coordpy.persistent_latent_v5`,
`coordpy.multi_hop_translator_v3`,
`coordpy.mergeable_latent_capsule`,
`coordpy.deep_proxy_stack_v4`,
`coordpy.ecc_codebook_v5`,
`coordpy.long_horizon_retention_v5`,
`coordpy.branch_merge_memory_v3`,
`coordpy.corruption_robust_carrier`,
`coordpy.transcript_vs_shared_arbiter_v2`,
`coordpy.uncertainty_layer`, and the composition module at
`coordpy.w53_team`, reachable only via explicit import; the
released v0.5.20 wheel's public surface is byte-for-byte
unchanged. See
[`RESULTS_W53_PMCRLOS.md`](RESULTS_W53_PMCRLOS.md) for the
full result note.

**Post-W51 research milestone — W52 Quantised Persistent Multi-Hop Latent Coordination**

The next research step after W51 (W52) introduces the
**Quantised Persistent Multi-Hop Latent Coordination
(QPMHLC)** layer: eight orthogonal capsule-native advances
on top of W51's six. (i) A two-layer **stacked persistent
latent state V4** with a learned signal skip-link that
preserves the turn-0 signal through mid-sequence
distractors. (ii) A **multi-hop quad-backend translator**
with chain-length-3 transitivity loss and a learned
disagreement-weighted arbitration that beats naive
equal-weight arbitration under per-edge confidence
calibration. (iii) An `L=8` **deep proxy stack V3** with
**role-conditioned KV banks** + **per-layer residual gate**.
(iv) A **three-level quantised codebook V4**
(`K1=32 × K2=16 × K3=8 = 4096` codes) + learned adaptive
budget gate — achieves ≥ 14 bits/visible-token at full emit.
(v) A **three-headed long-horizon reconstruction V4**
(causal + branch + cycle) at `max_k=12` with a degradation
curve probe across `k ∈ {1..24}`. (vi) A **branch/cycle
memory V2** with trainable merge + evict heads and joint
`(branch, cycle)` pages. (vii) A new **role-graph
conditioned cross-role transfer** module with per-edge
direction-dependent projections. (viii) A
**transcript-vs-shared-state matched-budget comparator** —
the first capsule-native ablation that compares transcript
truncation against shared-latent encoding under a fixed
visible-token budget. 26 new disjoint envelope failure modes
(cumulative trust boundary = **393 modes across W22..W52**).
W52 is held outside the stable SDK contract — it ships at
`coordpy.persistent_latent_v4`,
`coordpy.multi_hop_translator`,
`coordpy.deep_proxy_stack_v3`,
`coordpy.quantised_compression`,
`coordpy.long_horizon_retention_v4`,
`coordpy.branch_cycle_memory_v2`,
`coordpy.role_graph_transfer`,
`coordpy.transcript_vs_shared_state`, and the composition
module at `coordpy.w52_team`, reachable only via explicit
import; the released v0.5.20 wheel's public surface is
byte-for-byte unchanged. See
[`RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md`](RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md)
for the full result note.

**Post-W50 research milestone — W51 Persistent Cross-Backend Latent Coordination**

The next research step after W50 (W51) introduces the
**Persistent Cross-Backend Latent Coordination (PXBLC)**
layer: six orthogonal capsule-native advances on top of W50's
five. (i) A trainable **GRU-style persistent shared latent
state V3** that survives across turns, branches, and roles
via an update gate `z_t = sigmoid(W_z · [s_{t-1}; x_t])`
plus a content-addressed cross-role mixer with a per-role
view + learned blend. (ii) A **triple-backend translator**
over three backend tags `(A, B, C)` with direct translators
`A→B`, `A→C`, `B→C` plus a **transitivity loss** that
penalises disagreement between `A→C` and `A→B→C`. (iii) An
`L=6` **deep proxy stack V2** with **branch-specialised
heads** + **cycle-specialised heads** + per-layer learned
temperature. (iv) A **hierarchical adaptive compression V3**
with a coarse `K1=32` codebook + per-cluster fine `K2=16`
sub-codebooks plus a degradation-curve probe across
decreasing token budgets — achieves ≥ 12 bits/visible-token
at full emit. (v) A **two-headed long-horizon
reconstruction V3** (causal + branch) at `max_k=8` with a
degradation curve probe across `k ∈ {1..16}`. (vi) A
**branch/cycle-specialised memory head** with separate
per-branch and per-cycle pages plus learned cross-branch
consensus + cross-cycle merger. 24 new disjoint envelope
failure modes (cumulative trust boundary = **367 modes
across W22..W51**). W51 is held outside the stable SDK
contract — it ships at `coordpy.persistent_shared_latent`,
`coordpy.cross_backend_translator`,
`coordpy.deep_proxy_stack_v2`,
`coordpy.hierarchical_compression`,
`coordpy.long_horizon_retention`,
`coordpy.branch_cycle_memory`, and the composition module at
`coordpy.w51_team`, reachable only via explicit import; the
released v0.5.20 wheel's public surface is byte-for-byte
unchanged. See
[`RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md`](RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md)
for the full result note.

**Post-W48 research milestone — W49 Multi-Block Cross-Bank Coordination**

The next research step after W48 (W49) introduces the
**Multi-Block Cross-Bank Coordination (MBCC)** layer: the first
capsule-native CoordPy layer where `L_p`-stacked proxy
transformer blocks (each with multi-head attention + position-
wise feed-forward + residual scale) sit on top of **role-
conditioned multi-bank pseudo-KV** (one bank per role plus a
shared team bank), with reads aggregated by a learned **bank-mix
gate** and writes routed by a learned **bank-router**, slot
evictions decided by a trainable **eviction policy** (replacing
W48's FIFO), a separate **retention head** that answers a
binary "was this fact stored?" question against the multi-bank
read, a **dictionary codebook** (`K`-prototype) that quantises
the latent-control payload to a packed `LATENT_CTRL_V2` block
with strictly more structured bits per visible token than W48,
and a **content-addressed `SharedLatentCapsule` per turn** whose
value is the trained projection of the prior turn's multi-block
output and whose chain is recoverable from the envelope chain
alone. Each turn binds a per-turn `CrammingWitness` recording
structured-bits / visible-token frontier. Carries forward
W48's 22-mode verifier surface — **cumulative trust boundary
across W22..W49 = 323 enumerated failure modes**. W49 is held
outside the stable SDK contract — it ships at
`coordpy.multi_block_proxy` and is reachable only via explicit
import; the released v0.5.20 wheel's public surface is byte-for-
byte unchanged. See
[`RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md`](RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md)
for the full result note.

**Post-W46 research milestone — W47 Autograd Manifold Stack**

The next research step after W46 (W47) introduces the **Autograd
Manifold Stack (AMS)** layer: the first capsule-native CoordPy
layer where the manifold controller is **trained end-to-end by
autograd SGD/Adam** rather than stage-wise closed-form ridge.
Nine trainable, content-addressed components — all in pure
Python with no NumPy/JAX/PyTorch dependency: (i) a reverse-mode
`Variable` autograd engine with finite-difference gradient
checks; (ii) a trainable multi-layer tanh manifold stack; (iii)
a trainable rank-r LoRA-style role adapter; (iv) a trainable
K-prototype dictionary with soft-assignment cross-entropy; (v)
a trainable QKV memory head over the W46 bank; (vi) a trainable
packed-control serializer (4 sigmoid emit gates); (vii) a
pure-Python Adam optimiser; (viii) a content-addressed
`TrainingTraceWitness` sealing seed + optimiser config + loss
history + grad-norm history + final params CID; (ix) an
`AutogradManifoldTeam` orchestrator that reduces to
`ManifoldMemoryTeam.run` byte-for-byte under trivial config.
W47 **closes** the `W46-C-AUTOGRAD-DEEP-STACK` carry-forward
under the explicit "pure-Python reverse-mode AD + Adam" reading.
W47 is held outside the stable SDK contract — it ships at
`coordpy.autograd_manifold` and is reachable only via explicit
import; the released v0.5.20 wheel's public surface is
byte-for-byte unchanged. See
[`RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md`](RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md)
for the full result note.

**Post-W45 research milestone — W46 Manifold Memory Controller**

The next research step after W45 (W46) introduces the **Manifold
Memory Controller (MMC)** layer: the first capsule-native
CoordPy layer where the gating policy is shaped by a **bounded,
content-addressed memory of past turns**, runs through a
**multi-layer fitted controller stack**, applies **rank-r role
adapters**, encodes channel features into a **learned
dictionary basis**, emits a **packed multi-token
`MANIFOLD_CTRL` model-facing control surface**, and reuses a
**deterministic shared-prefix capsule** across consecutive
turns. Seven content-addressed components — all closed-form-
fittable in pure Python: (i) multi-layer learned controller
stack fitted stage-wise on layer-wise residuals; (ii) bounded
manifold memory bank with capsule-CID provenance per entry;
(iii) causally-masked time-attention readout (cosine-similarity
softmax over admissible past entries); (iv) rank-r LoRA-style
role adapter stack; (v) learned K-prototype dictionary basis
with bijective encode/decode; (vi) `MANIFOLD_CTRL` packed
multi-line control block carrying `route + conf + p +
layer_logits + mem_attn + dict_idx + mem_summary`; (vii)
shared-prefix capsule derived from the first-N prior-output
SHAs that emits byte-identical prefix bytes across turns. W46
is held outside the stable SDK contract — it ships at
`coordpy.manifold_memory` and is reachable only via explicit
import; the released v0.5.20 wheel's public surface is
byte-for-byte unchanged. See
[`RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md`](RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md)
for the full result note.

## Where to read next

If you want to use CoordPy:

* [`README.md`](../README.md) — product landing page, install, CLI,
  stable surface
* [`examples/01_quickstart.py`](../examples/01_quickstart.py) —
  smallest stable provider-backed agent-team example
* [`examples/02_quant_desk.py`](../examples/02_quant_desk.py) —
  curated four-role quant-desk preset against a real backend
* [`examples/03_replay_and_audit.py`](../examples/03_replay_and_audit.py) —
  dump a sealed manifest, replay it on a fresh backend, re-hash
  the new chain
* [`examples/`](../examples/) — full ladder + bundled scenario file

If you want to understand the released result:

* [`docs/RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`](RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md) — final
  release result note
* [`docs/SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`](SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md) —
  pre-committed success bar
* [`docs/RESEARCH_STATUS.md`](RESEARCH_STATUS.md) — current claims and
  status
* [`docs/THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md) — theorem and
  conjecture index
* [`docs/HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md) — claim
  boundary
* [`papers/context_as_objects.md`](../papers/context_as_objects.md) —
  main paper draft

## Historical research record

Everything below is preserved as the per-milestone audit trail. Use
the sections above for current onboarding; use the table below when
you need milestone-by-milestone history.

> **Current canonical reading.** The active scientific and product
> position is captured by a small set of files; everything else is
> historical record under [`archive/`](archive/).
>
> | Topic                                | Live doc                                                           |
> | ------------------------------------ | ------------------------------------------------------------------ |
> | One-pass orientation                 | this file (`docs/START_HERE.md`)                                   |
> | What is true *now*                   | [`RESEARCH_STATUS.md`](RESEARCH_STATUS.md)                         |
> | Theorem-by-theorem status            | [`THEOREM_REGISTRY.md`](THEOREM_REGISTRY.md)                       |
> | What may be claimed (do-not-overstate) | [`HOW_NOT_TO_OVERSTATE.md`](HOW_NOT_TO_OVERSTATE.md)               |
> | Run-boundary capsule formalism (W3)  | [`CAPSULE_FORMALISM.md`](CAPSULE_FORMALISM.md)                     |
> | Team-boundary capsule formalism (W4) | [`CAPSULE_TEAM_FORMALISM.md`](CAPSULE_TEAM_FORMALISM.md)           |
> | Long-running master plan             | [`context_zero_master_plan.md`](context_zero_master_plan.md)       |
> | Two-Mac MLX runbook                  | [`MLX_DISTRIBUTED_RUNBOOK.md`](MLX_DISTRIBUTED_RUNBOOK.md)         |
> | Post-W54 research milestone (W55)    | [`RESULTS_W55_DTDA_LOS.md`](RESULTS_W55_DTDA_LOS.md) |
> | Pre-committed success bar (W55)      | [`SUCCESS_CRITERION_W55_DEEP_TRUST_LATENT_OS.md`](SUCCESS_CRITERION_W55_DEEP_TRUST_LATENT_OS.md) |
> | Post-W53 research milestone (W54)    | [`RESULTS_W54_DMD_LOS.md`](RESULTS_W54_DMD_LOS.md) |
> | Pre-committed success bar (W54)      | [`SUCCESS_CRITERION_W54_DEEP_MERGE_LATENT_OS.md`](SUCCESS_CRITERION_W54_DEEP_MERGE_LATENT_OS.md) |
> | Post-W52 research milestone (W53)    | [`RESULTS_W53_PMCRLOS.md`](RESULTS_W53_PMCRLOS.md) |
> | Pre-committed success bar (W53)      | [`SUCCESS_CRITERION_W53_PMCRLOS.md`](SUCCESS_CRITERION_W53_PMCRLOS.md) |
> | Post-W51 research milestone (W52)    | [`RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md`](RESULTS_W52_QUANTISED_PERSISTENT_MULTI_HOP.md) |
> | Pre-committed success bar (W52)      | [`SUCCESS_CRITERION_W52_QUANTISED_PERSISTENT_MULTI_HOP.md`](SUCCESS_CRITERION_W52_QUANTISED_PERSISTENT_MULTI_HOP.md) |
> | Post-W50 research milestone (W51)    | [`RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md`](RESULTS_W51_PERSISTENT_LATENT_COORDINATION.md) |
> | Pre-committed success bar (W51)      | [`SUCCESS_CRITERION_W51_PERSISTENT_LATENT_COORDINATION.md`](SUCCESS_CRITERION_W51_PERSISTENT_LATENT_COORDINATION.md) |
> | Post-W49 research milestone (W50)    | [`RESULTS_W50_CROSS_BACKEND_LATENT_COORDINATION.md`](RESULTS_W50_CROSS_BACKEND_LATENT_COORDINATION.md) |
> | Pre-committed success bar (W50)      | [`SUCCESS_CRITERION_W50_CROSS_BACKEND_LATENT_COORDINATION.md`](SUCCESS_CRITERION_W50_CROSS_BACKEND_LATENT_COORDINATION.md) |
> | Post-W48 research milestone (W49)    | [`RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md`](RESULTS_COORDPY_W49_MULTI_BLOCK_PROXY.md) |
> | Pre-committed success bar (W49)      | [`SUCCESS_CRITERION_W49_MULTI_BLOCK_PROXY.md`](SUCCESS_CRITERION_W49_MULTI_BLOCK_PROXY.md) |
> | Post-W47 research milestone (W48)    | [`RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md`](RESULTS_COORDPY_W48_SHARED_STATE_PROXY.md) |
> | Pre-committed success bar (W48)      | [`SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md`](SUCCESS_CRITERION_W48_SHARED_STATE_PROXY.md) |
> | Post-W46 research milestone (W47)    | [`RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md`](RESULTS_COORDPY_W47_AUTOGRAD_MANIFOLD.md) |
> | Pre-committed success bar (W47)      | [`SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md`](SUCCESS_CRITERION_W47_AUTOGRAD_MANIFOLD.md) |
> | Post-W45 research milestone (W46)    | [`RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md`](RESULTS_COORDPY_W46_MANIFOLD_MEMORY.md) |
> | Pre-committed success bar (W46)      | [`SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md`](SUCCESS_CRITERION_W46_MANIFOLD_MEMORY.md) |
> | Post-W44 research milestone (W45)    | [`RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md`](RESULTS_COORDPY_W45_LEARNED_MANIFOLD.md) |
> | Pre-committed success bar (W45)      | [`SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md`](SUCCESS_CRITERION_W45_LEARNED_MANIFOLD.md) |
> | Post-W43 research milestone (W44)    | [`RESULTS_COORDPY_W44_LIVE_MANIFOLD.md`](RESULTS_COORDPY_W44_LIVE_MANIFOLD.md) |
> | Pre-committed success bar (W44)      | [`SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md`](SUCCESS_CRITERION_W44_LIVE_MANIFOLD.md) |
> | Post-release research milestone (W43) | [`RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md`](RESULTS_COORDPY_W43_PRODUCT_MANIFOLD.md) |
> | Pre-committed success bar (W43)      | [`SUCCESS_CRITERION_W43_PRODUCT_MANIFOLD.md`](SUCCESS_CRITERION_W43_PRODUCT_MANIFOLD.md) |
> | Latest release milestone (SDK v3.43 final)   | [`RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md`](RESULTS_COORDPY_W42_ROLE_INVARIANT_SYNTHESIS.md) |
> | Pre-committed success bar (SDK v3.43)| [`SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md`](SUCCESS_CRITERION_W42_ROLE_INVARIANT_SYNTHESIS.md) |
> | Previous milestone (SDK v3.42 RC2)   | [`RESULTS_COORDPY_W41_INTEGRATED_SYNTHESIS.md`](RESULTS_COORDPY_W41_INTEGRATED_SYNTHESIS.md) |
> | Pre-committed success bar (SDK v3.42)| [`SUCCESS_CRITERION_W41_INTEGRATED_SYNTHESIS.md`](SUCCESS_CRITERION_W41_INTEGRATED_SYNTHESIS.md) |
> | Previous milestone (SDK v3.41 RC1)   | [`RESULTS_COORDPY_W40_RESPONSE_HETEROGENEITY.md`](RESULTS_COORDPY_W40_RESPONSE_HETEROGENEITY.md) |
> | Pre-committed success bar (SDK v3.41)| [`SUCCESS_CRITERION_W40_RESPONSE_HETEROGENEITY.md`](SUCCESS_CRITERION_W40_RESPONSE_HETEROGENEITY.md) |
> | Previous milestone (SDK v3.40)       | [`RESULTS_COORDPY_W39_MULTI_HOST_DISJOINT_QUORUM.md`](RESULTS_COORDPY_W39_MULTI_HOST_DISJOINT_QUORUM.md) |
> | Pre-committed success bar (SDK v3.40)| [`SUCCESS_CRITERION_W39_MULTI_HOST_DISJOINT_QUORUM.md`](SUCCESS_CRITERION_W39_MULTI_HOST_DISJOINT_QUORUM.md) |
> | Previous milestone (SDK v3.39)       | [`RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md`](RESULTS_COORDPY_W38_DISJOINT_CONSENSUS_REFERENCE.md) |
> | Pre-committed success bar (SDK v3.39)| [`SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md`](SUCCESS_CRITERION_W38_DISJOINT_CONSENSUS_REFERENCE.md) |
> | Previous milestone (SDK v3.38)       | [`RESULTS_COORDPY_W37_CROSS_HOST_BASIS_TRAJECTORY.md`](RESULTS_COORDPY_W37_CROSS_HOST_BASIS_TRAJECTORY.md) |
> | Pre-committed success bar (SDK v3.38)| [`SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md`](SUCCESS_CRITERION_W37_CROSS_HOST_BASIS_TRAJECTORY.md) |
> | Previous milestone (SDK v3.37)       | [`RESULTS_COORDPY_W36_HOST_DIVERSE_TRUST_SUBSPACE.md`](RESULTS_COORDPY_W36_HOST_DIVERSE_TRUST_SUBSPACE.md) |
> | Pre-committed success bar (SDK v3.37)| [`SUCCESS_CRITERION_W36_HOST_DIVERSE_TRUST_SUBSPACE.md`](SUCCESS_CRITERION_W36_HOST_DIVERSE_TRUST_SUBSPACE.md) |
> | Previous milestone (SDK v3.36)       | [`RESULTS_COORDPY_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`](RESULTS_COORDPY_W35_TRUST_SUBSPACE_DENSE_CONTROL.md) |
> | Pre-committed success bar (SDK v3.36)| [`SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md`](SUCCESS_CRITERION_W35_TRUST_SUBSPACE_DENSE_CONTROL.md) |
> | Previous milestone (SDK v3.35)       | [`RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md`](RESULTS_COORDPY_W34_LIVE_AWARE_MULTI_ANCHOR.md) |
> | Pre-committed success bar (SDK v3.35)| [`SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md`](SUCCESS_CRITERION_W34_LIVE_AWARE_MULTI_ANCHOR.md) |
> | Previous milestone (SDK v3.34)       | [`RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md`](RESULTS_COORDPY_W33_TRUST_EWMA_TRACKED.md) |
> | Pre-committed success bar (SDK v3.34)| [`SUCCESS_CRITERION_W33_TRUST_EWMA_TRACKED.md`](SUCCESS_CRITERION_W33_TRUST_EWMA_TRACKED.md) |
> | Previous milestone (SDK v3.33)       | [`RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md`](RESULTS_COORDPY_W32_LONG_WINDOW_CONVERGENT.md) |
> | Pre-committed success bar (SDK v3.33)| [`SUCCESS_CRITERION_W32_LONG_WINDOW_CONVERGENT.md`](SUCCESS_CRITERION_W32_LONG_WINDOW_CONVERGENT.md) |
> | Previous milestone (SDK v3.32)       | [`RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md`](RESULTS_COORDPY_W31_ONLINE_CALIBRATED_GEOMETRY.md) |
> | Pre-committed success bar (SDK v3.32)| [`SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md`](SUCCESS_CRITERION_W31_ONLINE_CALIBRATED_GEOMETRY.md) |
> | Previous milestone (SDK v3.31)       | [`RESULTS_COORDPY_W30_CALIBRATED_GEOMETRY.md`](RESULTS_COORDPY_W30_CALIBRATED_GEOMETRY.md) |
> | Pre-committed success bar (SDK v3.31)| [`SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md`](SUCCESS_CRITERION_W30_CALIBRATED_GEOMETRY.md) |
> | Previous milestone (SDK v3.30)       | [`RESULTS_COORDPY_W29_GEOMETRY_PARTITIONED.md`](RESULTS_COORDPY_W29_GEOMETRY_PARTITIONED.md) |
> | Pre-committed success bar (SDK v3.30)| [`SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md`](SUCCESS_CRITERION_W29_GEOMETRY_PARTITIONED.md) |
> | Previous milestone (SDK v3.29)       | [`RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md`](RESULTS_COORDPY_W28_ENSEMBLE_VERIFIED_MULTI_CHAIN.md) |
> | Previous milestone (SDK v3.28)       | [`RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md`](RESULTS_COORDPY_W27_MULTI_CHAIN_PIVOT.md) |
> | Previous milestone (SDK v3.27)       | [`RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md`](RESULTS_COORDPY_W26_CHAIN_PERSISTED_FANOUT.md) |
> | Previous milestone (SDK v3.26)       | [`RESULTS_COORDPY_W25_SHARED_FANOUT.md`](RESULTS_COORDPY_W25_SHARED_FANOUT.md) |
> | Previous milestone (SDK v3.25)       | [`RESULTS_COORDPY_W24_SESSION_COMPACTION.md`](RESULTS_COORDPY_W24_SESSION_COMPACTION.md) |
> | Previous milestone (SDK v3.24)       | [`RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md`](RESULTS_COORDPY_W23_CROSS_CELL_DELTA.md) |
> | Previous milestone (SDK v3.23)       | [`RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md`](RESULTS_COORDPY_CAPSULE_LATENT_HYBRID.md) |
> | Previous milestone (SDK v3.22)       | [`RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md`](RESULTS_COORDPY_MULTI_ORACLE_ADJUDICATION.md) |
> | Previous milestone (SDK v3.21)       | [`RESULTS_COORDPY_OUTSIDE_INFORMATION.md`](RESULTS_COORDPY_OUTSIDE_INFORMATION.md) |
> | Previous milestone (SDK v3.20)       | [`RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md`](RESULTS_COORDPY_DECEPTIVE_AMBIGUITY.md) |
> | Previous milestone (SDK v3.19)       | [`RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md`](RESULTS_COORDPY_RELATIONAL_DISAMBIGUATOR.md) |
> | Previous milestone (SDK v3.18)       | [`RESULTS_COORDPY_LIVE_COMPOSITION.md`](RESULTS_COORDPY_LIVE_COMPOSITION.md) |
> | Previous milestone (SDK v3.17)       | [`RESULTS_COORDPY_COMPOSED_REAL_LLM.md`](RESULTS_COORDPY_COMPOSED_REAL_LLM.md) |
> | Previous milestone (SDK v3.16)       | [`RESULTS_COORDPY_ATTENTION_AWARE.md`](RESULTS_COORDPY_ATTENTION_AWARE.md) |
> | Previous milestone (SDK v3.15)       | [`RESULTS_COORDPY_PRODUCER_AMBIGUITY.md`](RESULTS_COORDPY_PRODUCER_AMBIGUITY.md) |
> | Previous milestone (SDK v3.14)       | [`RESULTS_COORDPY_OPEN_WORLD_NORMALIZATION.md`](RESULTS_COORDPY_OPEN_WORLD_NORMALIZATION.md) |
> | Previous milestone (SDK v3.13)       | [`RESULTS_COORDPY_REAL_LLM_MULTI_ROUND.md`](RESULTS_COORDPY_REAL_LLM_MULTI_ROUND.md) |
> | Previous milestone (SDK v3.12)       | [`RESULTS_COORDPY_MULTI_ROUND_DECODER.md`](RESULTS_COORDPY_MULTI_ROUND_DECODER.md) |
> | Previous milestone (SDK v3.11)       | [`RESULTS_COORDPY_BUNDLE_DECODER.md`](RESULTS_COORDPY_BUNDLE_DECODER.md) |
> | Pre-committed success bar (SDK v3.13)| [`SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md`](SUCCESS_CRITERION_MULTI_AGENT_CONTEXT.md) |
> | Previous milestone (SDK v3.10)       | [`RESULTS_COORDPY_MULTI_SERVICE_CORROBORATION.md`](RESULTS_COORDPY_MULTI_SERVICE_CORROBORATION.md) |
> | Previous milestone (SDK v3.9)        | [`RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md`](RESULTS_COORDPY_CROSS_ROLE_CORROBORATION.md) |
> | Previous milestone (SDK v3.8)        | [`RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md`](RESULTS_COORDPY_CROSS_ROLE_COHERENCE.md) |
> | Previous milestone (SDK v3.7)        | [`RESULTS_COORDPY_SCALE_VS_STRUCTURE.md`](RESULTS_COORDPY_SCALE_VS_STRUCTURE.md) |
> | Repo top-level                       | [`../README.md`](../README.md), [`../ARCHITECTURE.md`](../ARCHITECTURE.md), [`../CHANGELOG.md`](../CHANGELOG.md) |
> | Historical record (read-only)        | [`archive/`](archive/) — pre-CoordPy theory, older CoordPy milestones, sprint prompts |
