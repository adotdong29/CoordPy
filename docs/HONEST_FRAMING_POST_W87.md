# Honest framing — post-W87

> **Canonical, mandatory honesty surface for any claim that
> follows from the W87 closure of meta-#49.**  Last touched
> 2026-05-22 after meta-#4 (W79-era) and meta-#49 (post-W83)
> were both closed on GitHub with all 24 + 16 sub-issues
> CLOSED.
>
> If this file disagrees with any other doc about whether we
> *solved multi-agent context*, this file is right and the
> other file is stale.

## The headline claim we are explicitly NOT making

**We did not solve context in multi-agent systems.**

What we did: closed the entire post-W83 blocker backlog
(meta-#49: 5/5 P0 + 8/8 P1 + 8/8 P2 + 3/3 P3 = 24/24
sub-issues) and the predecessor backlog (meta-#4: 16/16
sub-issues), with real, re-verifiable, audit-chained evidence
at every step and documented honest limits at every closure.
**That is the substrate to attack the problem.  It is not
the problem solved.**

This file is the mandatory honesty surface every reader,
paper draft, README claim, demo, or external pitch must
align with.

## What we shipped (and that's real)

This is genuine progress and worth respecting.

* Working multi-modal substrate with real open-weight VLM
  hidden-state reads on Colab Pro A100-40GB (LLaVA-1.5-7B
  at bf16) and on local CPU at fp32 (Moondream-2, 1.87 B
  params).  See `docs/RESULTS_W87_MULTI_MODAL_V1.md`.
* Content-addressed audit chain that re-verifies offline,
  with **one safety property (Merkle inclusion) mechanically
  proved in Lean 4** with zero `sorry` / zero `admit`.  See
  `docs/RESULTS_W87_FORMAL_VERIFICATION_V1.md`.
* Real BFT (Ed25519 + PBFT V1), differential privacy (Laplace
  / Gaussian + budget tracker), MPC (Shamir + Pedersen +
  Schnorr), schema evolution, drift detection, multi-tenancy
  isolation, GPU-deterministic substrate, event-graph GC —
  every one passes its own offline verifier.  See
  `docs/RESULTS_W86_P2_CLOSURES.md`.
* Deployable gateway with OTLP / Prometheus / structured-log
  observability, stdlib-only, optional opentelemetry-sdk
  bridge, sample Grafana dashboard.  See
  `docs/RESULTS_W87_OBSERVABILITY_V1.md`.
* Docker-compose multi-host substrate that mints HMAC keys,
  partitions, heals, and replays.  See
  `docs/RESULTS_W86_REAL_DISTRIBUTED.md`.
* Frontier-scale substrate coupling on Llama-3.1-8B-Instruct,
  OLMoE-1B-7B-Instruct (real MoE, 64 experts × top-8),
  LLaVA-1.5-7B-hf, Qwen2.5-Coder-1.5B.  See
  `docs/RESULTS_W86_FRONTIER_CLOSURE.md`.

That is a strong substrate for attacking multi-agent context.

## What we explicitly did NOT solve

The mandatory carry-forward limits.  Every one of these is
load-bearing for honesty.

### 1. The fair-budget multi-agent superiority claim is NOT empirically established

**Carry-forward: `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN`**

On the one published-benchmark fair head-to-head we ran
(HumanEval, 3 seeds × 30 problems × `meta/llama-3.1-8b-instruct`):

* B (CoordPy multi-agent + executor-as-critic) mean pass@1
  = **71.1 %** beats A0 (stock single-shot) = 63.3 %
  by **+7.8 pp**.  This is the literal #28 DoD bullet
  ("strict improvement on at least one published metric vs
  the stock harness") — met honestly.
* B mean pass@1 = 71.1 % **loses** to A1 (K=5 first-pass-
  with-visible-test-filter, same compute budget) = 80.0 %
  by **−8.9 pp**.

The Reflexion / Self-Debug literature's "multi-agent + executor
beats self-consistency at same budget" claim **does NOT
replicate at this scale on this configuration**.

This alone is enough to say we have not solved multi-agent
context.  A substrate that loses to self-consistency at the
same compute budget on the one fair benchmark we ran has not
demonstrated multi-agent superiority — only single-shot
superiority.

The W85 GSM8K negative result reinforces this: on
N=20 × 3-seed Llama-3.1-8B, B (71.7 %) lost to BOTH A0 (75 %)
and A1 (81.7 %).  `W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-
BEAT-SELF-CONSISTENCY-CAP` is the matching carry-forward.

### 2. Frontier-scale evidence is single-host, not multi-machine WAN

**Carry-forward: `W86-L-MULTI-HOST-DISTRIBUTED-V1-DOCKER-BRIDGE-CAP`**

The #29 closure is a 3-container docker-compose topology on a
docker bridge network.  Containers are separate hosts in the
kernel-namespace + hostname + filesystem-layer + virtual-NIC
sense, but they share the host's hardware clock and Linux
kernel.  The V2 multi-physical-machine-on-WAN path is
documented in `docs/PLAN_W86_29_REAL_MULTI_HOST.md` and is
**NOT** closed.

### 3. Cross-modal reasoning is NOT shown load-bearing-better than single-modal

**Carry-forward: `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP`**

The W87 multi-modal substrate runs three modalities (text +
image + code) through a composed pipeline with a single cross-
modality Merkle root.  Every per-modality payload is real and
the audit chain spans them.  **But:** V1 is read-only across
modalities.  We never demonstrated that a vision + code + text
team outperforms any single-modal team on a published
benchmark.  Cross-modal injection (e.g. swapping an image
embedding mid-LLM-forward to affect code generation) is V2.

### 4. Formal verification covers ONE property at ONE fanout

**Carry-forwards: `W87-L-FORMAL-MERKLE-FANOUT-2-CAP`,
`W87-L-FORMAL-MERKLE-HASH-AXIOMATIC-CAP`,
`W87-L-FORMAL-MERKLE-PYTHON-INCLUSION-PATH-SHAPE-CAP`**

We formally verified bidirectional Merkle inclusion in Lean 4
for perfect binary trees (fanout = 2), with collision-
resistance of `hashPair` as the single explicit axiom.  The
Python implementation uses fanout = 4 by default, and the
inclusion-path shape Python returns (parent CIDs only) is
poorer than the (direction, sibling) form the Lean proof
covers.  **Other safety properties — replay-byte-identity,
consensus safety under f Byzantine adversaries, cross-tenant
no-leakage — are NOT formally verified.**  The DoD bar
("at least one safety property formally verified") is
honestly met; the stronger bar ("the entire safety story is
mechanically checked") is V2/V3.

### 5. Multi-agent benchmarks are largely toy / few-agent / synthetic

Most W6X–W79 multi-agent regimes are synthetic harness
benchmarks designed by us, with 2–5 agents.  W86 #28 is the
first real published benchmark (HumanEval) with a real
executor, and it shipped the negative same-budget result above.
We have **not** validated CoordPy at 10+ agents over weeks of
real interaction, with real users, on tasks the literature
recognises as load-bearing.

### 6. The substrate is closed at V1 — many V2+ frontiers remain

Per-issue carry-forwards (sampled, not exhaustive):

* `W86-L-QUANT-INT8-NEEDS-BNB-CUDA-COLAB-CAP` — quantized
  runtime V1 covers bf16; int8 with bitsandbytes is V2.
* `W86-L-MOE-SUBSTRATE-V1-TOPK-RESTORE-CAP` — MoE substrate V1
  restores top-K expert IDs + gate weights; full router-logits
  distribution restore is V2.
* `W86-L-MOE-SUBSTRATE-V1-SINGLE-GPU-CAP` — MoE substrate V1
  is single-GPU; expert-parallel multi-GPU is V2.
* `W86-L-LAGRANGIAN-V1-FLOOR-NOT-RESPECTED-ALONE-CAP` — #34
  closure is via Lagrangian + projection fallback; pure
  Lagrangian alone violates the safety floor.
* `W87-L-OBSERVABILITY-V1-PROMETHEUS-TEXT-CAP` — observability
  uses Prometheus text exposition v0.0.4; protobuf is V2.
* `W87-L-OBSERVABILITY-V1-HEAD-SAMPLING-CAP` — V1 is head
  sampling only; tail-based is V2.
* `W87-L-MULTI-MODAL-V1-AST-PYTHON-ONLY-CAP` — AST-aware code
  axis covers Python; other languages need tree-sitter (V2).

The full carry-forward set lives in `docs/THEOREM_REGISTRY.md`
under each milestone's `*-L-*` entries; the rule-set lives in
`docs/HOW_NOT_TO_OVERSTATE.md`.

## What would actually constitute "solving" it

The honest bar.  Until these are met, the right framing is
**"we built the substrate to attack the problem honestly"**,
not **"we solved the problem"**.

1. **Beat the strongest same-budget baseline on multiple
   diverse published benchmarks**, not just HumanEval-vs-
   stock-single-shot.  Target set: SWE-bench, AgentBench,
   MMVet (multi-modal), HotPotQA / MuSiQue (multi-hop QA),
   GAIA (agentic).
2. **Real long-horizon memory utility** measured over weeks
   or months of agent interaction, not over our in-repo
   bounded-window synthetic regimes.
3. **Production multi-tenant deployments under adversarial
   load**, not docker-compose + synthetic regimes.
4. **Cross-modal reasoning shown to be load-bearing-better**
   than single-modal (vision + code + text agents
   outperforming any single-modal ensemble on a published
   benchmark).
5. **Multi-physical-machine WAN substrate**, not docker
   bridge.
6. **Literature convergence** — other research groups
   independently validating the approach as best-in-class for
   the problem.
7. **The full safety story formally verified**, not one
   property at one fanout.

Until at least items 1–4 are met, claims of "solved" are
overstatement.

## Where this fits in the canonical doc map

* `README.md` — never repeats "solved" claims; describes
  capabilities + carry-forwards.
* `docs/RESEARCH_STATUS.md` — single-source-of-truth for
  *what is true now*; links here for the "did we solve it?"
  framing.
* `docs/THEOREM_REGISTRY.md` — every named theorem-style
  claim, with status (proved / mechanically-proved / proved-
  conditional / mechanically-checked / empirical / conjectural
  / retracted).
* `docs/HOW_NOT_TO_OVERSTATE.md` — the canonical do-not-
  overstate rules per milestone.  References this file as the
  post-W87 frame.
* `docs/AUDIT_POST_W83_BLOCKERS.md` — per-issue audit
  verdicts (closed vs partial vs blocked-on-hardware).
* `docs/HONEST_FRAMING_POST_W87.md` (this file) — the
  meta-honesty surface: what we did, what we did not, what
  "solving" would actually mean.

## The next research wave

The next meta issue, when the next wave hits, should target
the empirical bar in §"What would actually constitute
'solving' it" — specifically items 1 (multi-benchmark same-
budget multi-agent superiority) and 4 (cross-modal reasoning
load-bearing-better than single-modal).

Until that meta issue is closed with the same discipline as
meta-#49, claims of having solved multi-agent context are
overstatement and must be rejected by any reader of this
file.
