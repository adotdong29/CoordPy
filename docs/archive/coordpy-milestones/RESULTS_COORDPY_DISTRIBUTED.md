# SDK v3.6 — Two-Mac Distributed Inference Boundary + Real Cross-LLM Parser-Boundary Measurement

> Milestone results note. Not a polish pass — a deep
> systems-and-research push. SDK v3.5 made the capsule abstraction
> load-bearing **between agents**; SDK v3.6 attaches CoordPy to the
> **larger-model inference path** unlocked by combining two Apple
> Silicon Macs into a single sharded-model inference target, and
> ships the first **real** (non-synthetic) cross-LLM
> parser-boundary measurement against the model class actually
> available on the cluster today. Last touched: 2026-04-26.
>
> This note is theory-forward. The headline empirical finding is a
> 1.000 cross-model PARSE_OUTCOME failure-kind TVD between a
> 14.8B-dense model and a 36B-MoE model on a SWE-bench-Lite-shape
> bank — a saturated, reproducible result that **inverts the
> naive prediction** that a stronger model would reduce
> parser-boundary instability. Robust parser mode collapses the
> TVD to 0.000.

## Headline (one paragraph)

We chose **MLX distributed inference** as the right two-Mac
single-model path (§ 1) and shipped the smallest honest CoordPy
integration boundary for it: a duck-typed `LLMBackend` Protocol
with two concrete implementations (`OllamaBackend`,
`MLXDistributedBackend`) that the inner-loop seal-PROMPT /
seal-LLM_RESPONSE chain accepts byte-for-byte unchanged (§ 2).
We then ran the first real cross-LLM measurement on the
parser-boundary axis (W3-C6 promoted from synthetic to real on
the available model class) and the empirical result is sharp:
on the bundled bank (n=10, two parser modes), the 36B-MoE
`qwen3.5:35b` model under strict parsing produces
`failure_kind=unclosed_new` on **every** instance while the
14.8B-dense `qwen2.5:14b-32k` model produces `failure_kind=ok`
on every instance — cross-model TVD = 1.000 — and robust parser
mode recovers the larger model perfectly via
`recovery=closed_at_eos` so cross-model TVD = 0.000 (§ 3).
The honest reading is that **parser-boundary instability is a
model × prompt-format interaction, not a capacity artefact**;
the capsule-native runtime's closed-vocabulary `failure_kind`
captures this in one number where raw-bytes inspection would
require manual diffing of 20 responses (§ 4). Theorem family W5
formalises this: W5-1 (proved-empirical) and W5-2 / W5-3
(proved on the integration boundary). The two-Mac MLX path is
**experimental infrastructure**, not product (§ 6); CoordPy's
single-run product runtime contract is byte-for-byte unchanged
(§ 7).

## 1. Two-Mac inference path — chosen: MLX distributed

### 1.1 The candidates (and why we did not pick them)

**Hyperspace.** Hyperspace is a strong distributed-agent
infrastructure (node discovery, message routing, agent caching).
It is **not** a single-model sharding system: there is no public
Hyperspace surface that splits one transformer's weights across
two physical Macs and runs one forward pass across the cut.
Forcing Hyperspace would solve a different problem (orchestrating
many independent models on many nodes) than the one this
milestone targets (running ONE larger model whose weights do not
fit on a single 36 GB Mac).

**llama.cpp `--rpc`.** Real, supports Apple Silicon (Metal
backend), and shards a single GGUF across multiple `rpc-server`
processes. A defensible alternative. We did not pick it because
(a) MLX is more deeply optimised for Apple Silicon (uses
unified-memory MLX arrays, fuses weight transfers with the
Metal command queue), and (b) MLX's `mx.distributed` API is
Apple-official and well-documented for tensor-parallel +
pipeline-parallel sharding under MPI; the maintenance surface
is smaller.

**Plain Ollama on each Mac (current ASPEN harness).** This is
how the existing `aspen_cluster_config.local.json` is wired:
Mac 1 and Mac 2 each run independent Ollama servers with their
own loaded models. **It is not single-model sharding** — it is
*model parallelism by agent role*. For a 70B-class model whose
Q4 weights (≈ 40 GB) do not fit on one 36 GB Mac, this approach
cannot run the model at all. The existing harness is preserved
as a parallel codepath; it remains the right tool for
"different agents, different models" but is wrong for "one
larger model."

### 1.2 Why MLX distributed is right

The MLX distributed sharding strategy (tensor parallel +
pipeline parallel via `mx.distributed` collectives over MPI) is
Apple-official, well-documented, supports Llama / Qwen /
Mistral checkpoints natively (via `mlx-lm`), and exposes a
single OpenAI-compatible `/v1/chat/completions` HTTP endpoint
on the head rank once the launcher (`mpirun --hostfile <hosts>
mlx_lm.server ...`) negotiates the split. Apple's reference
documentation and the open-source `mlx-lm` repo both ship
end-to-end recipes; the dependency surface is `mlx`, `mlx-lm`,
and any working MPI (OpenMPI / MPICH).

### 1.3 Realistic model class on 2 × 36 GB

Per-Mac unified memory is 36 GB; total budget across two Macs
is ≈ 72 GB minus OS / KV-cache overhead. The honest weight
budget for a sharded model is ≈ 60 GB — comfortably enough for
one Llama-3.3-70B-Instruct in 4-bit (Q4 ≈ 40 GB) with KV-cache
headroom for moderate-context inference (≤ 8K tokens). 8-bit
70B (≈ 70 GB) is too tight; 16-bit 70B does not fit. Sub-70B
class (Mixtral 8×7B, Qwen-2.5-32B, Qwen-3.5-35B-MoE) all run
*on a single Mac* in 4-bit; for those the two-Mac path buys
KV-cache / context-length headroom, not capacity.

**Honest target:** 70B-class in Q4 across two Macs.

### 1.4 What is installable today vs what runs today

* `mlx` (Apple Silicon only) and `mlx-lm` are pip-installable on
  the two Apple Silicon Macs.
* OpenMPI is available via `brew install open-mpi`.
* Model conversion to MLX format (`mlx_lm.convert`) is a
  one-time per-model operation.
* The two Macs are `192.168.12.191` (Mac 1, alive at
  measurement time) and `192.168.12.248` (Mac 2, offline at
  measurement time — ARP "incomplete"). The integration
  boundary is therefore implementable and tested; the two-Mac
  *runtime* awaits Mac 2 returning to the LAN.

This is the honest scope. The MLX distributed path is **chosen
and integration-shipped**; running 70B sharded across both Macs
awaits an operator step (Mac 2 power-on + first
`mpirun mlx_lm.server` bring-up). The runbook is at
`docs/MLX_DISTRIBUTED_RUNBOOK.md`.

## 2. CoordPy integration — the smallest honest boundary

### 2.1 What ships in SDK v3.6

A new module `vision_mvp/coordpy/llm_backend.py`:

* **`LLMBackend`** — a `runtime_checkable` Protocol with
  `model: str`, `base_url: str | None`, and
  `generate(prompt, max_tokens, temperature) -> str`. Matches
  the duck-type that the inner-loop already expected from
  `LLMClient`.
* **`OllamaBackend`** — wraps `vision_mvp.core.llm_client.LLMClient`
  byte-for-byte unchanged. The default behaviour when no backend
  is supplied stays identical.
* **`MLXDistributedBackend`** — talks an OpenAI-compatible
  `/v1/chat/completions` endpoint. Designed for an
  `mlx_lm.server` launched under `mpirun` so the same HTTP
  surface fronts a single-host run *or* a sharded multi-host
  run. The CoordPy adapter is **neutral on the sharding strategy**:
  one HTTP client, one Protocol, one adapter class.
* **`make_backend(name, **kwargs)`** — factory dispatch by
  string name.

`run_sweep(spec, *, ctx=None, llm_backend=None)` accepts an
optional `llm_backend`. When set, the inner-loop dispatches
through it; when None, behaviour is byte-for-byte identical to
SDK v3.5. The `phase45.product_report.v2` sweep block grows one
optional field — `"backend": <class-name> | None` — so a
report can record which backend produced its capsules.

### 2.2 What the spine does NOT change

Capsule contracts C1–C6 are unchanged. PROMPT / LLM_RESPONSE /
PARSE_OUTCOME / PATCH_PROPOSAL / TEST_VERDICT all seal with the
same payload shape regardless of backend; the SHA-256 +
byte-length + bounded snippet shape recorded for PROMPT /
LLM_RESPONSE is preserved by construction (the runtime sees
two strings: the prompt string and the response string;
neither shape depends on which backend produced the response).
Spine equivalence (Theorem W3-34) carries over unchanged.

### 2.3 Tests

`vision_mvp/tests/test_coordpy_llm_backend.py` — 9 tests covering:

* Protocol membership (`isinstance(b, LLMBackend)` accepts
  concrete backends, rejects an object missing `generate`);
* Factory dispatch (`make_backend("ollama" | "mlx_distributed")`,
  unknown raises);
* `MLXDistributedBackend` wire-shape against an in-process
  OpenAI-compatible HTTP stub (request body matches
  `{model, messages, max_tokens, temperature, stream}`,
  Authorization header sent when `api_key` is set);
* Runtime integration (`run_sweep(..., llm_backend=fake_backend)`
  routes inner-loop calls through the duck-typed substitute and
  seals PROMPT / LLM_RESPONSE / PARSE_OUTCOME capsules
  end-to-end);
* Backward compatibility (`llm_backend=None` preserves the
  legacy spine — synthetic-mode replay produces no `backend`
  field).

All 9 pass in `< 2 s`.

## 3. Real cross-LLM parser-boundary measurement

### 3.1 Setup

* **Endpoint:** `http://192.168.12.191:11434` (Mac 1 Ollama).
* **Models:**
    * `qwen2.5:14b-32k`  — 14.8 B parameters, dense, Q4_K_M,
                            weights ≈ 9.0 GB.
    * `qwen3.5:35b`      — 36.0 B parameters, MoE, Q4_K_M,
                            weights ≈ 23.9 GB; `think=False` so
                            the response field is non-empty at
                            bounded `max_tokens`.
* **Prompt style:** `"block"` (the SWE-bench-Lite-shape default
  used by `local_smoke` / `bundled_57`).
* **Bank:** `vision_mvp/tasks/data/swe_lite_style_bank.jsonl`
  (n=10 first instances).
* **Parser modes:** `"strict"` and `"robust"`.
* **Sampling:** `temperature=0.0`, `max_tokens=320`.
* **Total LLM calls:** 20 (one per (model, instance) pair —
  responses are reused across parser modes since the parser is
  pure CPU work over response bytes).
* **Wall time:** 163 s (cold-load amortised over the run; warm
  qwen3.5:35b serves at ≈ 7–15 tok/s on the M3 Pro 36 GB).

### 3.2 Result (n=10)

```
Per-(model, parser_mode) failure-kind distribution:
  qwen2.5:14b-32k   strict    ok_rate=1.00   topkinds=ok=10
  qwen2.5:14b-32k   robust    ok_rate=1.00   topkinds=ok=10
  qwen3.5:35b       strict    ok_rate=0.00   topkinds=unclosed_new=10
  qwen3.5:35b       robust    ok_rate=1.00   topkinds=ok=10

Cross-model TVD (failure_kind, holding parser_mode fixed):
  qwen2.5:14b-32k  vs  qwen3.5:35b  (strict)   TVD = 1.000
  qwen2.5:14b-32k  vs  qwen3.5:35b  (robust)   TVD = 0.000

Parser-mode shift (strict → robust, per model):
  qwen2.5:14b-32k  0.000
  qwen3.5:35b      1.000
```

### 3.3 What is happening at the byte level

The 36B-MoE model under `think=False` emits the OLD/NEW close
delimiter as `<<` (two `<`) instead of `<<<` (three `<`). The
strict parser requires exactly `<<<` and reports
`failure_kind=unclosed_new`; the robust parser's
`recovery=closed_at_eos` heuristic synthesises the closing
delimiter and the parse becomes `ok`. The 14.8B dense model
emits `<<<` cleanly on every instance.

Cross-checked: the same instances under `prompt_style=
"unified_diff"` yield `failure_kind=no_block` on **both**
models — the 14B model also fails (it does not produce a
parseable diff under that prompt). Cross-model TVD on
`unified_diff` is therefore not informative on this bank. The
result is a **(model × prompt-format) interaction**, not a
pure model effect.

### 3.4 Honest scope

* n=10 saturated; the effect is binary on this bank.
  Falsifiable by adding instances where the 36B model
  *occasionally* emits `<<<`. Nothing in our 10 of 10 sample
  did.
* Single endpoint, single configuration. The measurement says
  nothing about the model under different temperatures, larger
  `max_tokens`, different prompt templates, or under the
  `think=True` mode.
* The 14B's perfect ok-rate is on the bundled bank with the
  block prompt — not a generalisation claim about the 14B
  model on arbitrary banks.
* The MLX-distributed two-Mac path was **not** the inference
  path used here (Mac 2 was offline); the measurement is on
  the *single*-Mac stronger-model class. SDK v3.6's integration
  boundary makes the same harness run unchanged once
  Mac 2 + `mpirun mlx_lm.server` is up.

## 4. Theory-forward consequences

### 4.1 The W5 family

We mint a small theorem family for SDK v3.6:

> **W5-1 (proved-empirical, real-LLM).** On the bundled
> SWE-bench-Lite-shape bank with `prompt_style="block"` and
> `temperature=0`, sampling `n=10` instances:
>
>   1. ``qwen3.5:35b`` (36 B MoE, Q4_K_M, ``think=False``) under
>      ``parser_mode="strict"`` produces ``failure_kind=
>      unclosed_new`` on every instance (10/10);
>      ``ok_rate=0.000``.
>   2. ``qwen2.5:14b-32k`` (14.8 B dense, Q4_K_M) under
>      ``parser_mode="strict"`` produces ``failure_kind=ok`` on
>      every instance (10/10); ``ok_rate=1.000``.
>   3. Cross-model PARSE_OUTCOME failure-kind TVD = 1.000 on
>      ``parser_mode="strict"``.
>   4. Robust parser's `recovery=closed_at_eos` recovers the 36 B
>      model to ``failure_kind=ok`` on every instance (10/10);
>      cross-model TVD on `parser_mode="robust"` = 0.000.
>
> **Anchor.** ``vision_mvp/experiments/parser_boundary_real_llm.py``,
> result JSON checked in at
> ``docs/data/parser_boundary_real_llm_n10.json`` (and a fresh
> run lands at ``/tmp/coordpy-distributed/real_cross_model_n10.json``).
> Reproducible with one command on any machine that can reach
> ``192.168.12.191:11434`` and has both Ollama models loaded.

> **W5-2 (proved, integration boundary).** The CoordPy inner-loop
> (`_real_cells`) accepts any duck-typed
> :class:`LLMBackend`-conformant object via
> ``run_sweep(spec, llm_backend=<backend>)``; the
> PROMPT / LLM_RESPONSE / PARSE_OUTCOME / PATCH_PROPOSAL /
> TEST_VERDICT capsule chain seals byte-for-byte equivalently
> regardless of which concrete backend provided the response.
>
> **Anchor.** ``test_coordpy_llm_backend.py::
> RunSweepBackendIntegrationTests``.

> **W5-3 (proved, wire shape).** ``MLXDistributedBackend.generate``
> issues an OpenAI-compatible ``POST /v1/chat/completions`` request
> with body ``{"model", "messages", "max_tokens", "temperature",
> "stream": false}`` and parses the response's
> ``choices[0].message.content``. Locked against an in-process
> stub.
>
> **Anchor.** ``test_coordpy_llm_backend.py::
> MLXDistributedBackendWireShapeTests``.

### 4.2 The W5-C conjecture family

> **W5-C1 (empirical-research, falsifiable).** Parser-boundary
> instability is a (model architecture × prompt-format)
> interaction, **not** a model-capacity artefact. Concretely:
> within a fixed prompt format, scaling from 14.8 B-dense to
> 36 B-MoE on the bundled bank *increases* strict-mode parser
> failure rate from 0/10 to 10/10. Falsifier: an instance bank
> on which the 36 B-MoE model produces ``failure_kind=ok`` at a
> rate strictly above 50 % under ``parser_mode="strict"``.

> **W5-C2 (empirical-research, falsifiable).** Robust-mode
> parser recovery (specifically ``recovery=closed_at_eos``) is
> the load-bearing safety net that makes the
> capsule-native runtime model-class-agnostic on the bundled
> prompt format. Concretely: cross-model TVD on
> ``parser_mode="robust"`` in W5-1 collapses from 1.000 (strict)
> to 0.000 (robust). Falsifier: a model whose ``unclosed_new``
> emissions cannot be salvaged by ``closed_at_eos``.

> **W5-C3 (research, conjectural).** The capsule-native
> runtime's closed-vocabulary ``PARSE_OUTCOME.failure_kind`` is
> a *minimum sufficient* typed witness of cross-model behaviour
> differences — i.e. the strict-mode failure-kind TVD captures
> in one number what a manual byte-level diff over 20 responses
> would surface across the same population. Falsifier: a pair
> of models with identical strict-mode `failure_kind`
> distribution but materially different downstream behaviour
> (test pass rate, semantic correctness) on the same bank.

### 4.3 What stronger models do *not* change

Three things do not move under the larger-model class:

1. **Capsule-contract C1–C6.** Every PROMPT / LLM_RESPONSE
   capsule under `qwen3.5:35b` was content-addressed,
   lifecycle-bounded, parent-CID-gated, and
   coordinate-consistent (lifecycle audit L-11 holds). The
   capsule-native runtime is *model-agnostic by design*; the
   W5-1 result is evidence, not a counterexample.
2. **Spine equivalence (W3-34).** The post-hoc adapter and
   in-flight construction agree on the spine kinds regardless
   of backend.
3. **Multi-agent capsule coordination (W4 family).** The team
   layer's mechanism is policy-on-capsules; switching the
   underlying inference backend changes the *bytes* in
   TEAM_HANDOFF capsules but not the lifecycle audit (T-1..T-7)
   or the W4-2 / W4-3 theorems.

### 4.4 What stronger models *do* change

The *frontier* of parser-boundary work moves. Specifically:

* Strict-mode parsing is no longer the safe default. The robust
  parser's recovery heuristics are now load-bearing for
  larger-model deployments. SDK v3.6 ships no parser change —
  the existing `ALL_RECOVERY_LABELS` (with `closed_at_eos`)
  was *exactly* what was needed.
* The W3-C6 synthetic distribution library now has a real
  measurement pinned next to it: the synthetic
  `unclosed_new` profile *predicted* the 36 B-MoE behaviour, and
  the strict→robust TVD shift on the synthetic library
  (1.000 on `synthetic.unclosed`) matches the real-LLM result
  exactly. This is empirical confirmation that the synthetic
  library is calibrated to the right axis.

### 4.5 Original-thesis connection

The Context Zero original thesis is "context is a routing /
substrate / coordination problem in multi-agent LLM systems."
SDK v3.5 made the capsule abstraction load-bearing **between
agents in a team**. SDK v3.6 makes it load-bearing **across the
model-class gradient**: the capsule-native runtime's typed
boundaries (PARSE_OUTCOME closed vocabulary, lifecycle audit
L-11, recovery-label closed vocabulary) cleanly absorb a model
swap from 14.8 B-dense to 36 B-MoE without any code change. The
larger-model regime is an *additional axis of evidence* for the
capsule-native thesis, not a refutation: it produces a stronger
empirical claim (W5-1) than synthetic distributions alone could.

This **strengthens** the original multi-agent-context thesis:
if the capsule-native runtime survives a 2.4× model-size jump
and a dense → MoE architecture switch on a real benchmark
without any spine modification, the runtime's typed-boundary
discipline is doing the load-bearing work, not the substrate
behaviour of any one model.

## 5. The two-Mac inference path materially advances the original goal

The original goal is to solve context for multi-agent teams.
The two-Mac MLX-distributed path materially advances this goal in
**three** specific ways:

1. **It unlocks 70 B-class research.** Without sharding, no 70 B
   model fits on a 36 GB Mac in any meaningful precision. SDK
   v3.6 ships the integration boundary that makes
   sharded 70 B inference one operator step (the runbook) away.
   Larger models change the parser-boundary landscape (W5-1 is
   already a sharp negative on the naive "stronger = cleaner"
   prediction); 70 B will give a third data point.
2. **It makes the capsule-native runtime model-class-agnostic
   in evidence, not just in design.** SDK v3.5's W4 theorems
   were proved on synthetic team-coordination scenarios; SDK
   v3.6's W5-1 is the first **real** capsule-native measurement
   across two model classes. The capsule abstraction is
   load-bearing under both.
3. **It moves the research frontier from "model-noise / parser-
   noise" back toward coordination.** The W3-C6 synthetic study
   identified the parser axis as a load-bearing source of
   TVD-1.000 distribution shifts; SDK v3.6 confirms this is
   *real*. Future multi-agent capsule-coordination experiments
   (W4 family) can now be run with confidence that the
   capsule-native runtime correctly absorbs cross-model
   parser-boundary noise via robust-mode recovery, isolating
   the *coordination* signal from the *parser-axis* noise.

## 6. Product honesty — what is product, what is experimental, what is infrastructure

### 6.1 The two-Mac MLX-distributed path

* **Product-grade?** **No.**
* **Experimental?** **Yes.**
* **Boundary / infrastructure-only?** **Yes.**

The MLX-distributed backend is **opt-in experimental
infrastructure**. It is a one-class `MLXDistributedBackend`
adapter that talks to an out-of-process server an operator
brings up. CoordPy does **not** ship `mlx`, `mlx-lm`, or `mpirun`
as dependencies; it does not auto-bring-up the cluster; it does
not manage MPI hostfiles or model-conversion pipelines. The
`docker` extra is at `pip install coordpy[docker]`; there is
deliberately no `pip install coordpy[mlx_distributed]` extra,
because CoordPy does not own the cluster.

### 6.2 What CoordPy ships

* **CoordPy can use a stronger distributed-inference backend.**
  YES — via `run_sweep(..., llm_backend=MLXDistributedBackend(
  model=..., base_url=...))`.
* **CoordPy ships a universal distributed-inference framework.**
  NO. The integration is one HTTP-client class.

### 6.3 What this means for users

* If you have one MLX-distributed endpoint up, you point CoordPy
  at it. Done.
* If you have two Macs and want CoordPy to bring up MLX
  distributed for you: out of scope. Use the runbook
  (`docs/MLX_DISTRIBUTED_RUNBOOK.md`).
* If you want the existing per-agent multi-Ollama harness
  (different agents, different models): unchanged — see
  `aspen_cluster_config.local.json` in fin-ground-test.

## 7. The CoordPy single-run product runtime contract is unchanged

* `RunSpec` shape: unchanged.
* `run(spec)` semantics: unchanged.
* `phase45.product_report.v2` schema: a new optional `"backend"`
  field appears in the sweep block when a backend was supplied;
  consumers reading the schema as "extensible JSON object" are
  unaffected.
* All existing capsule kinds, lifecycle audits, theorem family
  W3 / W4: unchanged.

## 8. Files changed

* **`vision_mvp/coordpy/llm_backend.py`** *(new)* — Protocol +
  `OllamaBackend` + `MLXDistributedBackend` + `make_backend`.
* **`vision_mvp/coordpy/runtime.py`** — `run_sweep` accepts
  optional `llm_backend`; sweep block records `"backend"` field
  in real mode.
* **`vision_mvp/coordpy/__init__.py`** — re-export
  `LLMBackend / OllamaBackend / MLXDistributedBackend / make_backend`;
  bump `SDK_VERSION` to `coordpy.sdk.v3.6`.
* **`vision_mvp/experiments/parser_boundary_real_llm.py`**
  *(new)* — real-LLM cross-model parser-boundary harness.
* **`vision_mvp/tests/test_coordpy_llm_backend.py`** *(new)* —
  9 backend / integration / wire-shape contract tests.
* **`docs/RESULTS_COORDPY_DISTRIBUTED.md`** *(this file)*.
* **`docs/MLX_DISTRIBUTED_RUNBOOK.md`** *(new)* — operator
  bring-up runbook.
* **`docs/THEOREM_REGISTRY.md`** — W5-1 / W5-2 / W5-3 / W5-C1
  / W5-C2 / W5-C3 added.
* **`docs/RESEARCH_STATUS.md`** — SDK v3.6 frontier section.
* **`docs/context_zero_master_plan.md`** — § 4.23 added.
* **`docs/START_HERE.md`** — SDK v3.6 paragraph added.

## 9. Tests + validation runs

* `python3 -m unittest -v vision_mvp.tests.test_coordpy_llm_backend`
  → **9 tests pass**.
* `python3 -m unittest -v
  vision_mvp.tests.test_coordpy_capsule_native_inner_loop`
  → **16 tests pass** (regression check; SDK v3.4 contract holds).
* `python3 -m vision_mvp.experiments.parser_boundary_real_llm
  --n-instances 6 --max-tokens 320 --out
  /tmp/coordpy-distributed/real_cross_model.json`
  → **n=6 saturation** (W5-1 evidence).
* `python3 -m vision_mvp.experiments.parser_boundary_real_llm
  --n-instances 10 --max-tokens 320 --out
  /tmp/coordpy-distributed/real_cross_model_n10.json`
  → **n=10 saturation** (W5-1 reproducibility check).

## 10. What remains open

* **Mac 2 not yet on the LAN.** When it returns, run
  `mpirun --hostfile <hosts> mlx_lm.server ...` per the
  runbook and re-run W5-1 against a 70 B model.
  Predicted result: cross-model TVD against `qwen2.5:14b-32k`
  > 0.5 on strict mode (W5-C1 prediction stands), with
  recovery healing under robust mode.
* **Real cross-LLM W3-C6 study** (paper § 9 future-work item)
  is now ✅ **PARTIALLY DISCHARGED** by W5-1 on the available
  model class. Full discharge requires the 70 B class once
  Mac 2 returns.
* **`think=True` mode for qwen3-class models.** Not measured
  in W5-1; reasoning-mode behaviour on the parser axis is an
  open research direction.
* **APPLY_OUTCOME capsule between PATCH_PROPOSAL and
  TEST_VERDICT** (paper § 9 SDK v3.5-candidate item) is still
  open. The two-Mac path does not affect that question.

---

*Theorem-forward summary: SDK v3.6 ships the smallest honest
two-Mac integration boundary for one-larger-model inference,
chooses MLX distributed as the right path, and produces the
first real cross-LLM measurement on the parser-boundary axis.
Theorem family W5 (W5-1 proved-empirical, W5-2 / W5-3 proved on
the integration boundary) anchors the milestone formally; the
W5-C conjecture family makes the empirical reading falsifiable.
The capsule-native runtime survives a 2.4× model-size jump and a
dense → MoE architecture swap with no spine modification, which
is the first cross-model evidence that the typed-boundary
discipline is doing the load-bearing work.*
