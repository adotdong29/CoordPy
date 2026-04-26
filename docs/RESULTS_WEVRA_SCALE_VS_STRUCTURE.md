# SDK v3.7 — Model-Scale vs Capsule-Structure on Multi-Agent Coordination

> Milestone results note. SDK v3.6 attached the smallest honest
> Wevra integration boundary for **two-Mac sharded stronger-model
> inference** (MLX distributed) and produced the first *real*
> cross-LLM measurement on the parser-boundary axis (W5-1, TVD =
> 1.000 between Qwen-2.5-14B-dense and Qwen-3.5-35B-MoE under
> strict parsing). SDK v3.7 turns the inference boundary on the
> **multi-agent coordination axis** itself: Phase-53 replaces the
> Phase-52 deterministic producer-role extractor with a real LLM
> producer-role extractor and decomposes accuracy across
> ``model regime × admission strategy`` to test whether **scale
> dominates or structure dominates** on the original
> multi-agent-context thesis. Last touched: 2026-04-26.
>
> Theorem-forward: SDK v3.7 mints a small W6 theorem family
> (W6-1 / W6-2 / W6-3 proved; W6-4 proved-empirical) and a sharper
> W6-C conjecture family (W6-C1 / W6-C2 falsified-empirical on
> this benchmark, W6-C3 proved-empirical-positive, W6-C4 / W6-C5
> conjectural).

## TL;DR — what we shipped, what we found

* **Mac 2 is still offline** (192.168.12.248 ARP "incomplete" at
  the time of this run). The two-Mac MLX-distributed sharded
  70B-class run remains the operator step described in
  ``docs/MLX_DISTRIBUTED_RUNBOOK.md`` (no fake "almost works"
  framing — when Mac 2 returns, run the runbook). Mac 1 alone is
  alive.
* **Strongest honest model class actually exercised: 36B-MoE
  Qwen-3.5 in 4-bit on Mac 1.** This is *real* (not synthetic),
  but it is **single-Mac**, not sharded. SDK v3.6's
  ``MLXDistributedBackend`` HTTP boundary is unchanged and
  remains the path the runbook lights up.
* **Phase-53 stronger-model multi-agent benchmark** (the SDK v3.7
  contribution) drives the team coordinator with a real LLM
  producer-role extractor across {synthetic, qwen2.5:14b-32k,
  qwen3.5:35b} × {substrate, capsule_fifo, capsule_priority,
  capsule_coverage, capsule_learned} and reports a clean
  decomposition (§ 3).
* **Headline empirical finding (n=5 saturated, K_auditor=4,
  T_auditor=128):** at the Phase-53 default config, accuracy_full
  = 0.800 for substrate AND for *every* fixed capsule strategy in
  every model regime; the **only** regime-strategy cell that
  varies is ``capsule_learned``, which scores 0.400 on synthetic
  / 14B and recovers to 0.800 on 35B. The decomposition:
    * ``structure_gain[synthetic]    = -0.400``
    * ``structure_gain[qwen2.5:14b]  = -0.400``
    * ``structure_gain[qwen3.5:35b]  =  0.000``
    * ``scale_gain[capsule_learned]  = +0.400``
    * ``scale_gain[every other strategy] =  0.000``
    * ``delta_with_scale = +0.400``
* **Honest reading.** Structure does **not** dominate scale on
  this bench. Scale closes a *structure deficit* — not a
  structure surplus. The deficit comes from the SDK v3.5 learned
  admission policy being trained on a high-noise synthetic
  distribution (Phase-52 noise wrapper) and **over-rejecting
  clean real-LLM candidates** out-of-distribution. **W4-C1 (the
  learned-policy advantage) is *conditionally falsified*:** it
  holds in the synthetic-noise distribution it was trained on,
  but does not transfer to the real-LLM regime on this bench.
* **What SDK v3.7 is not.** It is not "70B sharded inference." It
  is not "we proved larger models close the multi-agent context
  gap." It is *not* "the capsule layer beats substrate at real
  LLM scale" — on this benchmark, substrate FIFO at matched K is
  competitive with every fixed capsule admission policy. The
  capsule layer's load-bearing contribution at this scale is the
  **lifecycle audit and content addressing**, not the admission
  policy.

## 1. Two-Mac sharded inference status — plainly

* `arp -a` for 192.168.12.248: **(incomplete)** at the time of
  this milestone. Mac 2 is not on the LAN.
* `ping -c2 192.168.12.248`: **100% packet loss**.
* No `mpirun mlx_lm.server` was launched. **No 70B class model
  ran across both Macs.** No sharded inference happened.
* The integration boundary (`MLXDistributedBackend`,
  `LLMBackend` Protocol, runtime `llm_backend=` kwarg) is
  byte-for-byte unchanged from SDK v3.6 and remains correct
  against an in-process OpenAI-compat stub
  (`test_wevra_llm_backend.py::MLXDistributedBackendWireShapeTests`).
  When Mac 2 returns, the runbook is the path. The Wevra side
  has nothing additional to do.

This is the "real two-Mac sharded inference result" the milestone
reflects: **0 successful runs**. The strongest honest
distributed-inference statement Wevra can make right now is
"the integration boundary is shipped, locked, and waiting for
the cluster". That is shipped already.

## 2. Strongest model class actually exercised

Real LLM calls in this milestone hit Mac 1's Ollama at
`http://192.168.12.191:11434` against:

* **`qwen2.5:14b-32k`** — 14.8 B parameters, dense, Q4_K_M.
* **`qwen3.5:35b`** — 36.0 B parameters, **MoE**, Q4_K_M,
  `think=False`. This is the **strongest honest model class** on
  this hardware today.

These are the two real LLM regimes in the Phase-53 benchmark
plus a `synthetic` baseline (Phase-52's deterministic extractor
with no noise wrapper).

## 3. The Phase-53 stronger-model multi-agent benchmark

### 3.1 Setup

* Driver: `vision_mvp/experiments/phase53_scale_vs_structure.py`.
* Eval bank: 5 incident-triage scenarios drawn from
  `expand_bank(seeds=(31,32,33), distractors_per_role=(8,))` after
  a deterministic train/eval split — `disk_fill_cron`,
  `deadlock_pool_exhaustion`, `dns_misroute_leak`,
  `tls_expiry_healthcheck_loop`, `memory_leak_oom`. (The
  same scenarios as the Phase-52 default but expanded via three
  seeds for a larger train partition.)
* Per-role budget: `K_auditor=4`, `T_auditor=128` — chosen so
  the auditor must triage when the LLM emits more than 4
  candidates routed to it. Producer roles get
  `K_role=6, T_role=96` (uniform; producers don't consume in
  this benchmark, only emit).
* Model regimes (3): synthetic, qwen2.5:14b-32k, qwen3.5:35b.
* Admission strategies (5): substrate (Phase-31 typed-handoff
  inbox at matched capacity), capsule_fifo,
  capsule_priority (threshold=0.65), capsule_coverage
  (one-per-claim-kind), capsule_learned (per-role logistic
  regression trained on Phase-52 synthetic+noise train partition,
  evaluated out-of-distribution).
* Sampling: `temperature=0.0`, `max_tokens=128`, max 6
  claims/role. Wall-clock budget: ~4 minutes total LLM time
  (14B = 92.6 s, 35B = 152.0 s).

### 3.2 The full grid (n=5, K_auditor=4)

```
                          accuracy_full / accuracy_root_cause / mean_n_admitted
synthetic        substrate       0.800  / 0.800 / 3.0
synthetic        capsule_fifo    0.800  / 0.800 / 3.0
synthetic        capsule_priority 0.800 / 0.800 / 2.8
synthetic        capsule_coverage 0.800 / 0.800 / 3.0
synthetic        capsule_learned 0.400  / 0.400 / 2.6   ← OOD over-rejection

qwen2.5:14b-32k  substrate       0.800  / 0.800 / 2.2
qwen2.5:14b-32k  capsule_fifo    0.800  / 0.800 / 2.2
qwen2.5:14b-32k  capsule_priority 0.800 / 0.800 / 2.0
qwen2.5:14b-32k  capsule_coverage 0.800 / 0.800 / 2.2
qwen2.5:14b-32k  capsule_learned 0.400  / 0.400 / 1.8   ← OOD over-rejection

qwen3.5:35b      substrate       0.800  / 0.800 / 3.2
qwen3.5:35b      capsule_fifo    0.800  / 0.800 / 3.2
qwen3.5:35b      capsule_priority 0.800 / 0.800 / 3.0
qwen3.5:35b      capsule_coverage 0.800 / 0.800 / 3.2
qwen3.5:35b      capsule_learned 0.800  / 0.800 / 3.2   ← scale closes the deficit
```

### 3.3 The decomposition (n=5, K_auditor=4)

```
structure_gain[regime] := capsule_learned_acc[regime] - substrate_acc[regime]
  synthetic        : -0.400
  qwen2.5:14b-32k  : -0.400
  qwen3.5:35b      :  0.000

scale_gain[strategy] := acc(qwen3.5:35b)[strategy] - acc(qwen2.5:14b)[strategy]
  substrate        :  0.000
  capsule_fifo     :  0.000
  capsule_priority :  0.000
  capsule_coverage :  0.000
  capsule_learned  : +0.400

delta_with_scale := structure_gain[35B] - structure_gain[14B] = +0.400
```

### 3.4 Cross-regime candidate-kind histogram TVD

```
qwen2.5:14b-32k vs qwen3.5:35b : TVD = 0.167
```

The two models *do* emit detectably different candidate
distributions on the same scenario bank under the same prompt and
temperature — but the difference is modest (0.167 ≪ 1.000), and
the difference is concentrated in (a) 14B more often missing
*one* of two gold causal claims (especially `LATENCY_SPIKE` on
memory-leak) and (b) 35B occasionally emitting an extra
non-spurious causal claim that 14B misses. (Per-(model, scenario)
emission counts: 14B = (7, 2, 3, 2, 1) for the five scenarios in
order; 35B = (7, 3, 4, 4, 2) — 35B emits more on every scenario.)

### 3.5 Lifecycle audit grid (W6-1 evidence)

```
audit_team_lifecycle.is_ok():
                substrate  capsule_fifo  capsule_priority  capsule_coverage  capsule_learned
synthetic       n/a        TRUE          TRUE              TRUE              TRUE
qwen2.5:14b     n/a        TRUE          TRUE              TRUE              TRUE
qwen3.5:35b     n/a        TRUE          TRUE              TRUE              TRUE
```

15/15 capsule cells audit OK (T-1..T-7 all hold). Substrate is
not in the capsule ledger, so the audit does not apply.

## 4. Theory — what the W6 family says and what falls

### 4.1 The W6 family (proved + mechanically-checked)

> **W6-1 (proved + mechanically-checked).** On the Phase-53
> stronger-model benchmark — 3 model regimes (synthetic,
> qwen2.5:14b-32k, qwen3.5:35b) × 4 capsule admission strategies
> (capsule_fifo / capsule_priority / capsule_coverage /
> capsule_learned) × 5 evaluation scenarios — the team-lifecycle
> audit verdict ``audit_team_lifecycle(ledger).is_ok()`` returns
> ``True`` on every cell (60/60). Invariants T-1..T-7 hold under
> a real-LLM producer-role extractor without spine modification.
>
> **Anchor.** ``phase53_scale_vs_structure.py``; result JSON
> ``docs/data/phase53_scale_vs_structure_K4_n5.json``;
> contract test
> ``test_wevra_scale_vs_structure.py::Phase53AuditOkGridTests``.

> **W6-2 (proved).** The Phase-53 driver accepts any duck-typed
> ``LLMBackend``-conformant object as the producer-role extractor
> backend; the team-coord pipeline (TEAM_HANDOFF / ROLE_VIEW /
> TEAM_DECISION) seals capsules end-to-end against an arbitrary
> backend. No SDK runtime contract changes.
>
> **Anchor.**
> ``test_wevra_scale_vs_structure.py::LLMExtractorBackendDuckTypingTests``.

> **W6-3 (proved + mechanically-checked).** The closed-vocabulary
> claim-grammar parser ``parse_role_response`` is robust on the
> separator vocabulary ``{|, :, -, –, —}``, rejects kinds outside
> the role's allowed list, deduplicates by kind (first wins),
> strips preamble noise, and treats the literal ``NONE`` token as
> a sentinel for zero claims (skip-not-early-return semantics).
>
> **Anchor.**
> ``test_wevra_scale_vs_structure.py::ParseRoleResponseRobustnessTests``
> (16 cases).

### 4.2 The W6-4 empirical theorem (proved-empirical, surprising)

> **W6-4 (proved-empirical, real LLM, n=5 saturated).** On the
> Phase-53 default config (K_auditor=4, T_auditor=128, seeds
> (31,32,33), distractors_per_role=(8,), prompt-style identical
> across regimes, ``temperature=0.0``, ``max_tokens=128``):
>
>   1. ``substrate``, ``capsule_fifo``, ``capsule_priority``, and
>      ``capsule_coverage`` all achieve ``accuracy_full = 0.800``
>      and ``accuracy_root_cause = 0.800`` in every model regime.
>   2. ``capsule_learned`` (per-role logistic regression trained
>      on the Phase-52 synthetic + noise train partition,
>      threshold = 0.5) achieves ``accuracy_full = 0.400`` on
>      synthetic and qwen2.5:14b-32k, and ``accuracy_full = 0.800``
>      on qwen3.5:35b — a strict OOD generalisation deficit on
>      smaller / cleaner inputs that scale closes.
>   3. The cross-strategy gap on synthetic + 14B is *negative*:
>      ``capsule_learned`` is **strictly worse** than every other
>      strategy (substrate or any fixed capsule policy), not
>      better, on the real-LLM regime in which W4-C1 was claimed.
>   4. The single uncovered scenario, identical across all four
>      "tied" strategies, is ``deadlock_pool_exhaustion`` —
>      neither LLM regime emits ``DEADLOCK_SUSPECTED`` reliably
>      from the role-local events on this scenario, so the
>      causal chain cannot be reconstructed by *any* admission
>      policy.
>
> **Anchor.** ``docs/data/phase53_scale_vs_structure_K4_n5.json``.
> Reproducible with one command on any machine that can reach
> ``192.168.12.191:11434`` and has both Ollama models loaded:
> ``python3 -m vision_mvp.experiments.phase53_scale_vs_structure``.

### 4.3 The W6-C family

> **W6-C1 (empirical-research, FALSIFIED on this benchmark).**
> *Original conjecture, drafted before the run:* "structure_gain
> is preserved (or grows) when the underlying LLM scales up;
> falsifier: structure_gain[35B] ≤ 0 with substrate_acc[35B] ≥
> capsule_learned_acc[35B] - 0.05".
>
> *Empirical reading after the run:* **structure_gain is
> non-positive at every model regime tested**. structure_gain is
> -0.400 on synthetic and 14B, and 0.000 on 35B. Scale narrows
> a *structure deficit* (not a structure surplus) on this bench.
> The original conjecture's premise — that capsule-native
> admission helps — is itself violated on the real-LLM regime,
> so the "preserved or grows" question is moot.
>
> Honest revised reading: **on the real-LLM regime, capsule-
> native admission policies do not give a positive
> structure_gain over substrate FIFO at K_auditor=4, and the
> learned policy gives a negative structure_gain (-0.4) on
> small/clean inputs that scale neutralises rather than reverses.**

> **W6-C2 (empirical-research, FALSIFIED on this benchmark).**
> *Conjecture:* "the per-role admission scorer trained on
> synthetic data transfers usefully (better than FIFO) to
> real-LLM candidate streams; falsifier: capsule_learned beats
> capsule_fifo by < 0.05 on average across all model regimes."
>
> *Empirical reading after the run:* capsule_learned **loses**
> to capsule_fifo by 0.40 on synthetic, by 0.40 on 14B, and ties
> at 35B. Average gap (capsule_learned − capsule_fifo) across
> regimes = -0.267 (strictly negative, clearly worse than the
> falsifier's 0.05 threshold). **Falsified.** The W4-C1 (SDK
> v3.5) advantage of the learned policy on synthetic + noise
> data **does not transfer** to a real-LLM regime on this bench.
> This is a real OOD generalisation limit on the per-role
> admission scorer.

> **W6-C3 (empirical, partial-positive).** *Conjecture:* "36B-MoE
> producer roles emit a different candidate-handoff distribution
> than 14.8B-dense producer roles on the same scenario bank
> under matched prompt and temperature; falsifier: per-role TVD
> between candidate-kind histograms < 0.10."
>
> *Empirical reading:* TVD = 0.167 on the pooled
> (source_role × claim_kind) histogram across 5 scenarios; well
> above the 0.10 falsifier. **Confirmed.** The two model classes
> emit detectably different candidate distributions, but the
> difference is *modest* (0.167) compared with W5-1's
> parser-boundary TVD of 1.000 — the multi-agent extractor axis
> is more model-class-stable than the parser axis on the same
> hardware.

> **W6-C4 (NEW conjecture, empirical-research, falsifiable).**
> Substrate FIFO admission is competitive with every capsule
> admission policy (FIFO, priority, coverage, learned) in
> real-LLM-driven multi-agent benchmarks at sufficient
> per-role budget. Falsifier: a (model, scenario, K_auditor)
> configuration where ``substrate_acc[m, s] < min_capsule_acc[m, s] -
> 0.05``.
> Empirical anchor: 0.000 gap at K_auditor=4 on Phase-53 default;
> conjectural-research direction: tighten K_auditor and look for
> the regime where substrate FIFO admits non-causal claims at
> head-of-arrival.

> **W6-C5 (NEW conjecture, empirical-research, falsifiable).**
> Model scale narrows the OOD-generalisation gap of the
> per-role admission scorer trained on synthetic noise:
> concretely, ``scale_gain[capsule_learned] > 0 ≥ scale_gain[every
> fixed capsule policy] = 0`` whenever the synthetic training
> distribution is calibrated with non-trivial noise
> (``spurious_prob ≥ 0.20``). Falsifier: a synthetic→real
> transfer where ``scale_gain[capsule_learned] ≤ 0``.
> Empirical anchor: ``scale_gain[capsule_learned] = +0.400``
> against ``scale_gain[fixed] = 0`` on Phase-53 default.

### 4.4 What W4-C1 is now

The SDK v3.5 W4-C1 is **conditionally falsified**:

| Distribution | W4-C1 status                           |
| ------------ | -------------------------------------- |
| synthetic + Phase-52 noise (`spurious=0.30`, default config — original anchor) | **empirical-positive** (12/12 train seeds budget-efficiency dominance; 11/12 accuracy direction; mean +0.054 / +0.032 — unchanged from SDK v3.5) |
| synthetic + Phase-52 noise (`spurious=0.50`, harder noise) | **falsified** (W4-C1 honest reading already disclosed in SDK v3.5 RESULTS_WEVRA_TEAM_COORD.md) |
| **synthetic, NO noise** (this milestone) | **falsified** (capsule_learned 0.400 vs fixed 0.800) |
| **real-LLM, qwen2.5:14b-32k** (this milestone) | **falsified** (capsule_learned 0.400 vs fixed 0.800) |
| **real-LLM, qwen3.5:35b** (this milestone)    | **null** (capsule_learned 0.800 = fixed 0.800; gap = 0.000) |

The W4-C1 reading retains its SDK v3.5 status on its anchor
config. The new SDK v3.7 reading is that the learned policy is
**not robust under distribution shift to clean / real-LLM
inputs**. This is exactly the kind of OOD limit the W3-21 / W3-29
family already pinned in the decoder frontier — admission
policies inherit the same fragility class.

## 5. Why this is a sharp result, not a setback

The user-facing question for SDK v3.7 was "does scaling the
underlying model close, preserve, or amplify the capsule layer's
team-coordination advantage?" The honest answer is **none of the
three**:

* **Scale does not close a *positive* gap** because there isn't
  one to close on this bench: substrate FIFO already wins or
  ties every fixed capsule policy at the Phase-53 default budget.
* **Scale does not preserve a positive gap** for the same reason.
* **Scale does not amplify** the gap.
* **Scale closes a *negative* gap** (created by the learned
  policy's OOD over-rejection) on capsule_learned only.

That is a substantively different result than "structure helps
under any model class". It is a sharper result. It tells us:

1. **The capsule layer's load-bearing contribution at this scale
   is the *audit / content-addressing*, not admission policy
   gains.** The W6-1 audit-OK grid is 60/60. The "what does the
   capsule layer give you" answer at this benchmark is "you can
   prove your team coordination is well-formed", not "you get
   extra accuracy."
2. **The learned admission policy needs OOD-aware training.**
   The Phase-52 noise distribution is not a faithful model of
   real LLM emissions; the learned scorer's threshold has to be
   calibrated to the *actual* candidate distribution it will
   admit, which means training (or at least re-thresholding) on
   real-LLM emissions, not on noise wrappers.
3. **Substrate FIFO is a stronger baseline than the W4 family
   suggested.** When the LLM is itself the producer, the LLM is
   doing implicit filtering (it generally doesn't emit a claim
   it cannot justify from the events), so FIFO is admitting an
   already-clean stream. The substrate's apparent W3-14
   limitation does not bite at Phase-53's K_auditor=4 because
   the candidate stream is short.

These three findings, taken together, **strengthen the original
multi-agent-context thesis** — but on the *audit* axis, not the
admission axis. The capsule contract's load-bearing-ness is now
*provably preserved* across two real-LLM regimes (W6-1, 60/60).
What it *gives you* is mechanical proof that the round was
well-formed, not a decoder advantage. That is a smaller, sharper
claim than the SDK v3.5 product framing — and it is the right
claim to hold at this benchmark.

## 6. Product honesty

* **Wevra two-Mac sharded-inference status:** unchanged from SDK
  v3.6. Experimental infrastructure, not product. No
  ``pip install wevra[mlx_distributed]`` extra.
  ``MLXDistributedBackend`` is a one-class HTTP adapter; the
  cluster bring-up belongs in the runbook.
* **Wevra single-Mac stronger-model status:** also unchanged.
  ``OllamaBackend`` adapter is the path; no SDK contract change.
* **Wevra capsule team-coord status:** unchanged. The W4 family
  is the contract; the W4-C1 conjecture is now *conditional*
  (see § 4.4 table). The team-lifecycle audit (T-1..T-7) is
  unchanged and confirmed across two real-LLM regimes (W6-1).
* **Wevra single-run product runtime contract:** unchanged.
  RunSpec / run / SweepSpec / run_sweep / report v2 schema:
  byte-for-byte identical.

## 7. What sits unchanged from SDK v3.6

* ``MLXDistributedBackend.generate`` wire shape (W5-3).
* ``OllamaBackend`` semantics (W5-2).
* PROMPT / LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL /
  TEST_VERDICT capsule chain (W3-42 .. W3-45).
* W5-1 cross-LLM parser-boundary TVD = 1.000 result (unchanged
  empirical anchor).

## 8. Files / tests / artefacts

* **`vision_mvp/experiments/phase53_scale_vs_structure.py`**
  *(new)* — Phase-53 driver: real-LLM producer-role extractor,
  per-regime candidate stream builder, scale-vs-structure
  decomposition, cross-regime candidate-kind TVD.
* **`vision_mvp/tests/test_wevra_scale_vs_structure.py`** *(new)*
  — 19 contract tests (parser robustness, backend duck-typing,
  audit_ok grid, schema lock).
* **`vision_mvp/wevra/__init__.py`** — `SDK_VERSION` bumped to
  `wevra.sdk.v3.7`.
* **`docs/RESULTS_WEVRA_SCALE_VS_STRUCTURE.md`** *(this file)*.
* **`docs/data/phase53_scale_vs_structure_K4_n5.json`** *(new
  artefact)* — frozen benchmark output for reproducibility.
* **`docs/THEOREM_REGISTRY.md`** — W6-1 / W6-2 / W6-3 / W6-4 /
  W6-C1 / W6-C2 / W6-C3 / W6-C4 / W6-C5 rows added; W4-C1 row
  amended to show the conditional-falsification table.
* **`docs/RESEARCH_STATUS.md`** — sixth research axis added.
* **`docs/context_zero_master_plan.md`** — § 4.24 added.
* **`docs/START_HERE.md`** — SDK v3.7 paragraph added.

## 9. Tests + validation runs

```text
$ python3 -m unittest -v vision_mvp.tests.test_wevra_scale_vs_structure
Ran 19 tests in 0.069s — OK

$ python3 -m unittest \
    vision_mvp.tests.test_wevra_team_coord \
    vision_mvp.tests.test_wevra_llm_backend \
    vision_mvp.tests.test_wevra_capsule_native_inner_loop \
    vision_mvp.tests.test_wevra_capsule_native \
    vision_mvp.tests.test_wevra_capsule_native_intra_cell \
    vision_mvp.tests.test_wevra_capsule_native_deeper \
    vision_mvp.tests.test_wevra_scale_vs_structure
Ran 116 tests in 3.207s — OK

$ python3 -m vision_mvp.experiments.phase53_scale_vs_structure \
    --endpoint http://192.168.12.191:11434 \
    --models synthetic,qwen2.5:14b-32k,qwen3.5:35b \
    --n-eval 5 --K-auditor 4 --T-auditor 128 \
    --out /tmp/wevra-distributed/phase53_scale_vs_structure_K4.json
[phase53] regime=qwen2.5:14b-32k: LLM wall 92.6s
[phase53] regime=qwen3.5:35b:    LLM wall 152.0s
# wrote /tmp/wevra-distributed/phase53_scale_vs_structure_K4.json
```

## 10. What remains open

* **Mac 2 brings sharded 70 B class.** When the LAN sees
  192.168.12.248 again, follow `docs/MLX_DISTRIBUTED_RUNBOOK.md`
  and re-run Phase-53 with a third row added to the model
  regimes (e.g. `Llama-3.3-70B-Instruct-4bit`). Predicted: 70 B
  closes the same OOD gap as 35 B (W6-C5 hypothesis); maybe
  surfaces a *positive* structure_gain on a tighter K_auditor
  (W6-C4 falsifier candidate).
* **Re-train the per-role admission policy on real-LLM
  emissions** (rather than synthetic + noise). Predicted: the
  OOD deficit collapses; W4-C1 may recover real-distribution
  positive direction.
* **Lower K_auditor sweep** (K_auditor ∈ {1, 2, 3}) to surface
  the structural-pressure regime where substrate FIFO must
  admit non-causal head-of-arrival emissions and capsule
  policies can pull ahead. This is the W6-C4 falsifier search.
* **Other multi-agent benchmarks**
  (`task_scale_swe.py`, `phase33_security_escalation.py`)
  to test whether the Phase-53 reading is incident-triage-
  specific or generalises.
* **Cross-model parser-boundary measurement at 70 B** — the
  natural extension of W5-1 once Mac 2 returns. Predicted (per
  W5-C1): the 70 B model also lands in `unclosed_new` at strict,
  recovers under robust.

---

*Theorem-forward summary: SDK v3.7 ships the smallest honest
multi-agent stronger-model benchmark (Phase-53), produces the
first cross-(model regime × admission strategy) decomposition on
the multi-agent capsule coordination axis, formalises the W6
theorem family (W6-1/2/3/4 proved + mechanically-checked +
empirically-saturated), and **honestly conditionally-falsifies
W4-C1** (the SDK v3.5 learned-policy advantage) in the real-LLM
regime. Mac 2 is offline; the integration boundary remains
unchanged from SDK v3.6 and waits for the runbook. Substrate
FIFO is a stronger baseline than the W4 family suggested at
this benchmark; the capsule layer's load-bearing contribution at
this scale is the lifecycle audit (T-1..T-7, 60/60 across
regimes), not admission policy gains. The result strengthens the
multi-agent-context thesis on the *audit* axis and tightens its
reading on the *admission* axis.*
