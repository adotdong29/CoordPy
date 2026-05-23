# Honest framing — post-W87

> **Canonical, mandatory honesty surface for any claim that
> follows from the W87 closure of meta-#49.**  Last touched
> 2026-05-22 to reflect the W89 post-W88 empirical superiority
> wave V2 results — **TWO carry-forwards RETIRED at 70B scale**:
> `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` and
> `W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`
> on the HumanEval prong; cross-modal prong still STAYS at all
> three model scales tested.  Prior touch was 2026-05-22 after
> the W88 wave (both negative / partial) and earlier after
> meta-#4 (W79-era) and meta-#49 (post-W83) were closed on
> GitHub with all 24 + 16 sub-issues CLOSED.
>
> If this file disagrees with any other doc about whether we
> *solved multi-agent context*, this file is right and the
> other file is stale.

## The headline claim we are explicitly NOT making

**We did not solve context in multi-agent systems.**

What we did:

* Closed the entire post-W83 blocker backlog (meta-#49: 5/5
  P0 + 8/8 P1 + 8/8 P2 + 3/3 P3 = 24/24 sub-issues) and the
  predecessor backlog (meta-#4: 16/16 sub-issues), with real,
  re-verifiable, audit-chained evidence at every step and
  documented honest limits at every closure.
* **W89 (2026-05-22): retired two empirical carry-forwards at
  70B scale** — at Llama-3.3-70B-Instruct on 3 seeds × 30
  HumanEval × K=5, the W88 sequential-reflexion B-pipeline
  strictly beats first-pass-among-K=5 self-consistency by
  +5.56 pp on the mean, with B winning on 2/3 seeds.  All 4
  pre-committed retirement bars met.  Audit chain re-derives
  offline 7/7 PASS.

**That is the substrate to attack the problem PLUS the first
empirical evidence in this programme of same-budget multi-
agent superiority over the strongest single-agent baseline
on a published benchmark.  It is still not the full problem
solved.**

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

### 1. The fair-budget multi-agent superiority claim is PARTIALLY established (at 70B scale on HumanEval only)

**Carry-forwards retired at 70B-HumanEval (W89):**
`W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN`,
`W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`

**Carry-forwards still active:**
`W89-L-HUMANEVAL-REFLEXION-V2-8B-CAP` (scopes the negative
result to 8B scale, where reflexion still loses);
`W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-BEAT-SELF-CONSISTENCY-CAP`
(different benchmark; not attacked by W89).

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

**W88 retry:** post-W87, we re-attacked this carry-forward with
a cleaner sequential-reflexion B-pipeline
(`coordpy.humaneval_reflexion_bench_v1`).  Same model, same
budget, same task subset discipline — different B-shape: 5
sequential model calls, each conditioned on the cumulative
history of prior candidates AND the actual executor stderr.
Every call is code-producing; W86's redundant judge call is
removed.  Result on 3 seeds × 30 problems × NIM Llama-3.1-8B
(bench Merkle `11997891e2b834fe…`, verifier 7/7 PASS):

* W88 B mean pass@1 = **71.1 %** (≈ W86 B); A1 mean = 74.4 %
  (≈ W86 A1 within sampling variance).
* W88 B − A1 = **−3.33 pp**.  The gap closed by **5.6 pp** vs
  W86's −8.9 pp.  The sign did NOT flip.  B beats A1 on 0/3
  seeds (ties on 2/3, loses 10 pp on seed 1).

W88 contributes the new carry-forward
`W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`
recording the second failure.  At W88 the W86 carry-forward
STAYS.  Two independent multi-agent B-shapes (executor-critic +
sequential-reflexion) both lose to first-pass-among-K=5
self-consistency at this scale on this model.

**W89 retry at 70B scale (2026-05-22) — RETIREMENT:**

The W88 sequential-reflexion B-pipeline ships unchanged; only
the NIM model id flips from `meta/llama-3.1-8b-instruct` to
`meta/llama-3.3-70b-instruct`.  Same K=5 budget; same task
subset per seed; same retry policy; same executor; same audit-
chain discipline (bench Merkle `977c213285995bd5…`; verifier
7/7 PASS):

* A0 mean pass@1 = 46.7 %  (single-shot 70B; an A0-T0 anomaly
  recorded as `W89-L-HUMANEVAL-REFLEXION-V2-A0-T0-ANOMALY-CAP`;
  does not affect B vs A1)
* A1 mean pass@1 = 85.6 %  (first-pass-among-K=5 at 70B; the
  strongest same-budget single-agent baseline)
* **B mean pass@1 = 91.1 %**  (sequential-reflexion-K=5 at 70B)
* **B − A1 = +5.56 pp**; B beats A1 on **2/3 seeds**
  (+13.3 / −3.3 / +6.7 pp per seed).
* All 4 pre-committed retirement bars in `RUNBOOK_W89.md`
  MET.

`W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` and
`W88-L-HUMANEVAL-REFLEXION-V1-A1-SAME-BUDGET-NOT-BEATEN-CAP`
are now both retired AT 70B scale on HumanEval.  The
honest scoped claim:

> **At Llama-3.3-70B-Instruct on 3 seeds × 30 HumanEval × K=5
> budget, sequential-reflexion conditioned on cumulative
> executor stderr strictly beats first-pass-among-K=5 self-
> consistency by +5.56 pp on the mean.  This is the first
> empirical demonstration in this programme of same-budget
> multi-agent superiority over the strongest single-agent
> baseline on a published benchmark.**

What this is NOT: a claim that multi-agent wins at all scales
(8B still loses), at all benchmarks (GSM8K still loses), or in
all forms (cross-modal splits still lose — see §3 below).
The W89 win is one published-benchmark win at frontier model
scale; the broader empirical bar still requires multi-benchmark
extension.

The 8B-scale negative carry-forward
`W89-L-HUMANEVAL-REFLEXION-V2-8B-CAP` records that smaller
instruction-tuned models do not exhibit this win — anyone
running this pipeline at 8B should expect B < A1, per the
W86 / W88 evidence.

The W85 GSM8K negative result is a different benchmark:
on N=20 × 3-seed Llama-3.1-8B, B (71.7 %) lost to BOTH A0
(75 %) and A1 (81.7 %).  `W85-L-GSM8K-BENCH-V1-MULTI-AGENT-DOES-NOT-
BEAT-SELF-CONSISTENCY-CAP` is the matching carry-forward and
stays unaffected by W89's HumanEval-specific retirement.
Whether the 70B-scale win extends to GSM8K is V2 work.

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

**Carry-forwards: `W87-L-MULTI-MODAL-V1-NO-CROSS-MODAL-INJECT-CAP`,
`W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`**

The W87 multi-modal substrate runs three modalities (text +
image + code) through a composed pipeline with a single cross-
modality Merkle root.  Every per-modality payload is real and
the audit chain spans them.  **But:** V1 is read-only across
modalities.  We never demonstrated that a vision + code + text
team outperforms any single-modal team on a published
benchmark.  Cross-modal injection (e.g. swapping an image
embedding mid-LLM-forward to affect code generation) is V2.

**W88 retry:** post-W87, we ran a published-benchmark fair
head-to-head with three arms on a synthesised HumanEval-Visual
corpus (`coordpy.cross_modal_code_bench_v1`, 3 seeds × 12
problems × NIM `meta/llama-3.2-11b-vision-instruct` +
`meta/llama-3.1-8b-instruct`, K=5 budget, bench Merkle
`37ac174e21cbe3f9…`, verifier 4/4 audit PASS).

**W89 cross-modal extension (2026-05-22):** the W88 single
configuration was extended with two retry runs to test whether
scaling models or hardening the image-load-bearing regime
retires the carry-forward.  Both stayed negative on the
team-organisation direction.  Three independent runs now
agree:

* **W88 V1 (11B-Vision + 8B-code, doctest_only)** — A0_text
  66.7 % / A1_vlm 86.1 % / B_cross 80.6 %.  B−A1 = −5.56 pp.
* **W89 P2 (90B-Vision + 8B-code, all_docstring)** — A0_text
  41.7 % / A1_vlm 86.1 % / B_cross 58.3 %.  B−A1 =
  **−27.78 pp** (split COLLAPSES when image is sole info
  source; text-only code-LM cannot verify image content).
* **W89 P3 (90B-Vision + 70B-code, doctest_only)** — A0_text
  33.3 % / A1_vlm 91.7 % / B_cross 86.1 %.  B−A1 =
  **−5.56 pp** (same gap as W88; scaling both models in
  proportion preserves the gap).

Cumulative result is split:

* **Image is empirically load-bearing across all three
  scales.**  W88 + W89 P2 + W89 P3 collectively show
  B_cross − A0_text margins of +13.9 pp, +16.7 pp, +52.8 pp —
  all exceed the +5.0 pp pre-committed threshold.  The W87
  multi-modal substrate IS carrying real load-bearing
  information.
* **The cross-modal team SPLIT is FALSIFIED at every scale
  tested.**  B_cross − A1_vlm = −5.6 / −27.8 / −5.6 pp at the
  three configurations.  Scaling the code-LM by ~9× (8B → 70B)
  and the VLM by ~8× (11B → 90B) does NOT flip the sign on
  doctest_only mode.  The `all_docstring` regime widens the
  gap to −27.8 pp because the text-only code-LM cannot
  recover from VLM extraction errors when the image is the
  sole information source.

W88 contributed `W88-L-CROSS-MODAL-CODE-V1-SPLIT-NOT-LOAD-BEARING-CAP`.
W89 adds `W89-L-CROSS-MODAL-SPLIT-MODEL-SCALE-INVARIANT-CAP`
(the gap is approximately invariant under model scale within
the tested range, in doctest_only mode) and
`W89-L-CROSS-MODAL-ALL-DOCSTRING-SPLIT-WORSE-CAP` (the gap
widens 5× in the harder image-strict regime).

**The W87 carry-forward STAYS at all three model-scale
configurations.  Scaling models alone cannot retire it.  The
next attempt must change the cross-modal architecture itself**
— e.g., keeping the VLM in the executor-feedback loop
(VLM-in-loop reflexion), parallel heterogeneous pool, or
deep cross-modal injection at substrate level.

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

### W88 → W89 contribution (2026-05-22)

The W88 wave V1 produced two negative / partial results.  The
W89 wave V2 retried with stronger model + harder image-load-
bearing regime and produced the FIRST RETIREMENT in this
programme:

**HumanEval prong (W89 wave V2): RETIREMENT.**  At Llama-3.3-
70B-Instruct scale on 3 seeds × 30 problems × K=5: B sequential
reflexion 91.1 % > A1 first-pass-K=5 85.6 % by **+5.56 pp**;
B beats A1 on 2/3 seeds; all 4 pre-committed retirement bars
met; audit chain re-derives 7/7 PASS.  The W86 / W88 HumanEval
carry-forwards retire AT 70B SCALE.  The 8B-scale evidence
remains canonical as `W89-L-HUMANEVAL-REFLEXION-V2-8B-CAP`.

**Cross-modal prong (W89 wave V2): STAYS NEGATIVE on
team-organisation.**  Image-load-bearing direction now
PROVEN at three model scales (B_cross − A0_text = +13.9 /
+16.7 / +52.8 pp).  Multi-agent split direction FALSIFIED at
all three scales (B_cross − A1_vlm = −5.6 / −27.8 / −5.6 pp).
Scaling models alone cannot retire.  The next attempt must
change architecture.

W89's contribution is the FIRST published-benchmark same-
budget multi-agent superiority demonstration at frontier model
scale.  What is now warranted:

* The Reflexion / Self-Debug literature replicates at 70B
  scale on HumanEval — under the W86/W88 anti-cheat
  discipline.
* Multi-agent CoordPy at frontier scale is no longer "an open
  question against the strongest same-budget single-agent
  baseline" on this one benchmark.  It beats the baseline.

What is still NOT warranted:

* "Multi-agent CoordPy beats single-agent at all scales" — the
  8B-scale negative result still holds.
* "Multi-agent CoordPy beats single-agent on all benchmarks" —
  W85 GSM8K still loses; W88/W89 cross-modal split still
  loses.
* "We solved multi-agent context" — the substrate-level work
  still requires multi-benchmark extension (MBPP+, SWE-bench,
  MATH, GSM8K-70B retry) and a working cross-modal architecture.

The next wave's task: extend the 70B-HumanEval win to
additional published benchmarks (MBPP+, MATH, GSM8K) AND find
a cross-modal architecture that actually wins (VLM-in-loop,
parallel heterogeneous pool, or substrate-level cross-modal
injection).
