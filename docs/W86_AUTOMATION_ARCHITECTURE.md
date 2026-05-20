# W86 — Automation architecture (runbook + map)

> **Audience:** new contributor or future-you re-running any
> W86 closure from scratch. This document is the canonical
> map of the W86 automation stack: every script, every
> benchmark, every closure path, and how they stitch together.
>
> **W86 milestone state at time of writing (commit
> `73a4c10`):** all five P0 substrate blockers (#25–#29)
> closed; seven of eight P1 blockers (#30, #32, #33, #34, #35,
> #36, #37) closed; #31 (MoE substrate) closure code is ready
> and pending one Colab Pro run.

## 1. The four hosts

W86 runs against **four distinct compute surfaces**, picked
per closure based on what the issue's DoD demands:

| Host | Used for | Cost | How to access |
|---|---|---|---|
| **Local Mac CPU** | Doc generation, audit-chain re-verification, lean-env unit tests, capacity / budget / Lagrangian / proof closures | Free | `python -m pytest`, direct script invocation |
| **Local Mac + Docker (colima)** | #29 multi-host distributed substrate (3-container topology on a docker bridge network) | Free | `docker compose -f docker/compose-w86-multi-host.yml up -d` |
| **NVIDIA NIM frontier text** | #28 HumanEval (multi-agent + executor-as-critic), W85 long-context task-success, GSM8K | Free with NVIDIA_API_KEY | `NIMFrontierTextRuntimeV1` |
| **Colab Pro browser (A100 / L4)** | #25 / #26 / #27 / #31 — anything that needs real GPU + open-weight model weights | Flat Pro subscription | Open `colab.research.google.com` URL pointing at a notebook in `scripts/` |

The hard rule from the user: **never bill GCP for compute on
this project.** Colab Pro browser is the GPU host; `gcloud
colab` (Colab Enterprise on Vertex AI) is OFF-LIMITS.

## 2. The closure map

Each closed P0/P1 maps to one or two scripts and a results
artefact. The artefacts are content-addressed JSON files; an
auditor re-verifies them offline with the matching verifier.

### P0 closures (meta-#49)

| Issue | Closure path | Driver | Verifier | Evidence on disk |
|---|---|---|---|---|
| **#25** Frontier-Scale Substrate Coupling | Colab Pro A100 + Llama-3.1-8B-Instruct in bf16 | `scripts/run_frontier_closure_w86.py` (subcommand: default; runs #25 + #26 + #27 in one driver) | `scripts/verify_w86_audit_chain.py` | `results/w86/w86_20260520T022828Z/25_substrate_coupling.json` |
| **#26** Live LLM Training of Composed Learned Memory | Same | Same driver | Same verifier | `results/w86/w86_20260520T022828Z/26_live_learned_memory.json` |
| **#27** Long-Context Live Eval ≥32k | Same (substrate axis); + W85 NIM (task-success axis) | Same driver | Same verifier | `results/w86/w86_20260520T022828Z/27_long_context_intercept.json` (substrate axis) + `results/w85/long_context_live_report_v2.json` (task axis) |
| **#28** Real-World Multi-Agent Task Benchmark | NIM Llama-3.1-8B + W84 PythonExecSandbox + 3 seeds × 30 HumanEval problems × 3 arms | `scripts/run_w86_humaneval_bench.py` | `scripts/verify_w86_humaneval_audit_chain.py` | `results/w86/humaneval/humaneval_bench_report.json` |
| **#29** Real Cross-Host Distributed Substrate | docker-compose 3-container topology (host-a, host-b, partition-proxy) on bridge network; W84 cross-process substrate code over real virtual NIC pairs | `scripts/run_w86_multi_host_bench.py` (one-shot orchestrator: mints HMAC keys → compose up → wait healthz → run bench → compose down) | `scripts/verify_w86_multi_host_audit_chain.py` | `results/w86/multi_host/multi_host_distributed_bench_report.json` + `_run2.json` |

### P1 closures (meta-#49)

| Issue | Closure path | Driver / module | Verifier | Evidence |
|---|---|---|---|---|
| **#30** Quantized-Runtime Substrate | bf16 tier via W86 transformers_runtime_v1 (CUDA + bf16 + skinny-trace); inherits #25 evidence | (re-uses #25 driver) | (re-uses #25 verifier) | `results/w86/w86_20260520T022828Z/25_substrate_coupling.json::replay_from_kv.precision_tier = "tier_bf16"` |
| **#31** MoE Substrate | Colab Pro A100 + OLMoE-1B-7B in bf16 (this round; pending run) | `scripts/run_w86_moe_substrate_closure.py` driven by `scripts/colab_moe_substrate_closure_w86.ipynb` | `scripts/verify_w86_moe_audit_chain.py` | `results/w86/moe/<TS>/moe_substrate_closure_report.json` (when run completes) |
| **#32** Streaming Substrate | W84 `streaming_substrate_intercept_v1.py` (per-token forward_stream + SSE + mid-stream injection) | `tests/test_w84_streaming_substrate.py` | (test-driven) | W84 test suite green |
| **#33** Tool Substrate | W84 `tool_call_substrate_v1.run_tool_substrate_team_bench_v1` + W86 HumanEval as production-style witness | Direct module invocation + `scripts/run_w86_humaneval_bench.py` | (test-driven + audit chain) | W84 5-agent bench output + W86 HumanEval audit chain |
| **#34** Online Learning Safety | W86 `lagrangian_with_projection_v1.run_lagrangian_with_projection_floor_recovery_v1` | Direct module invocation | (test-driven) | bench output (printed by tests) |
| **#35** Analytical Bounds | W84 `papers/proofs/w84_proof_*.md` (4 proofs, ≥3 bar) + theorem-registry entries | Read the proofs | (review) | `papers/proofs/` |
| **#36** Capacity Scaling | W86 `capacity_remediation_v2.run_capacity_remediation_v2_bench_v1` (cliff moves 112×) | Direct module invocation | (test-driven) | bench output |
| **#37** Hard Budget Enforcement | W86 `budget_enforced_composed_recovery_v1.run_budget_integration_head_to_head_v1` (composed-pipeline integration) | Direct module invocation | (test-driven) | bench output |

## 3. The audit-chain pattern

**Every closure ships a content-addressed JSON report and a
matching offline re-verifier.** The pattern is invariant
across closures:

1. The driver writes `<closure>_report.json` to `results/w86/.../`.
2. The report carries a top-level `report_cid` (or `bench_cid`)
   that hashes the canonical bytes of every field except
   itself.
3. Per-call sidecars (e.g. `humaneval_bench_report.calls.jsonl`,
   `phase_{a,b}.log`) ship next to the report so a third party
   can re-verify CIDs without re-running the model.
4. The verifier reads `<report>.json` + sidecar, re-derives
   every CID, prints `PASS` / `FAIL` per DoD bullet, exits 0
   iff every load-bearing bool is True.
5. CI (`tests/test_w86_*_audit_chain.py`) skip-gates on the
   report being present; when it is, the CI gate re-derives
   the CIDs and re-asserts the closure bools. This works
   without GPU / NIM / Docker.

### Hashing convention

```python
def _sha256(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"),
            default=str).encode("utf-8")).hexdigest()
```

`sort_keys=True` + the tightest separators is the canonical
form. The report's CID is computed over `{"kind":
"w86_<report_kind>_v1", "report": <report-dict-without-cid>}`.

## 4. Running each closure from scratch

### #25 / #26 / #27 frontier closure (Colab Pro A100, ~5 min)

```text
1. https://colab.research.google.com/github/adotdong29/CoordPy/blob/main/scripts/colab_frontier_closure_w86.ipynb
2. Runtime → Change runtime type → A100 GPU
3. 🔑 (left sidebar) → + Add new secret → name=hf_token,
   value=hf_xxxxxxxx (Meta-Llama-3.1 license accepted on HF);
   toggle Notebook access on
4. Runtime → Disconnect and delete runtime (clean start)
5. Open URL fresh → Runtime → Run all
6. Cells 5 + 6 run two PROCESS-ISOLATED phases:
   Phase A → closure_25 + closure_26 (full-trace runtime)
   30 s sleep (lets OS reclaim VRAM)
   Phase B → closure_27 (skinny-trace + SDPA, ≥32k tokens)
7. Drive save + zip download in cell 8
8. Local: `python scripts/verify_w86_audit_chain.py \
     --report results/w86/<TS>/frontier_closure_report.json`
```

Total wall ~5 min (77 s × 2 model loads + ~30 s benches).
Expected: `OVERALL: PASS` on the verifier, all 4 closure bars
green (conformance 10/12, replay 0.156 < 0.5, intercept moves
CID at short context AND at 32k).

### #28 HumanEval head-to-head (NIM, ~28 min)

```text
$ export NVIDIA_API_KEY=...  # from ~/.config/nim_api_key
$ W86_HE_N_PROBLEMS=30 W86_HE_N_SEEDS=3 \
    python scripts/run_w86_humaneval_bench.py
... writes results/w86/humaneval/humaneval_bench_report.json
$ python scripts/verify_w86_humaneval_audit_chain.py \
    --report results/w86/humaneval/humaneval_bench_report.json
```

Total wall ~28 min (~990 NIM calls at ~0.6 calls/sec).

### #29 docker-compose multi-host (~10 s + ~2 s bench)

```text
$ colima start --cpu 2 --memory 4    # if Docker daemon not up
$ python scripts/run_w86_multi_host_bench.py
[launch] mints HMAC keys → writes docker/.env.w86
[launch] docker compose -p coordpy_w86 -f \
    docker/compose-w86-multi-host.yml --env-file ... up -d --build
[launch] waits for host-a, host-b, partition-proxy /healthz
[launch] docker inspect for hostname + IP + network_id
[launch] runs run_multi_host_distributed_bench_v1 against
    127.0.0.1:18080 (host-a), 18081 (host-b), 19000 (proxy)
[launch] writes results/w86/multi_host/multi_host_distributed_bench_report.json
[launch] always: docker compose down -v + delete .env.w86
$ python scripts/verify_w86_multi_host_audit_chain.py \
    --report results/w86/multi_host/multi_host_distributed_bench_report.json
```

### #31 MoE substrate (Colab Pro A100, ~5 min) — NEW THIS ROUND

```text
1. https://colab.research.google.com/github/adotdong29/CoordPy/blob/main/scripts/colab_moe_substrate_closure_w86.ipynb
2. Runtime → Change runtime type → A100 GPU
3. 🔑 Same `hf_token` secret as #25–#27. OLMoE is NOT a
   gated repo, so the token is only used to avoid HF rate
   limits.
4. Runtime → Disconnect and delete runtime
5. Open URL fresh → Runtime → Run all
6. Driver runs the closure end-to-end:
   - probe model is MoE
   - load OLMoE-1B-7B in bf16 (model is 7 B / 1.3 B active /
     8 experts top-2)
   - forward + capture per-layer routing
   - replay-from-KV; compare last-token logits at bf16 tier
   - intercept hidden state on a MoE block; verify CID moves
   - check routing is deterministic across two forwards
7. Drive save + zip download
8. Local: python scripts/verify_w86_moe_audit_chain.py \
     --report results/w86/moe/<TS>/moe_substrate_closure_report.json
```

Expected: `OVERALL: PASS` on the verifier with the four #31
load-bearing bools green:
- `forward_routing_captured = True` (per-layer routing
  snapshotted via per-block forward hooks)
- `replay_with_routing_matches_forward_floor = True`
  (last-logits diff < bf16 tier tolerance 0.5)
- `moe_routing_is_load_bearing = True` (routing is captured
  and deterministic — the *negative* claim that proves
  routing is state, not freely synthesisable)
- `hidden_state_intercept_on_moe_block_moves_cid = True`

### #30 / #32 / #33 / #34 / #35 / #36 / #37 P1 closures (local CPU, <2 min combined)

All of these are local. Run the focused regression sweep:

```text
python -m pytest \
    tests/test_w86_budget_enforced_composed_recovery.py \
    tests/test_w86_lagrangian_with_projection.py \
    tests/test_w86_capacity_remediation_v2.py \
    tests/test_w86_moe_substrate.py \
    tests/test_w84_budget_enforcement.py \
    tests/test_w84_capacity_bench.py \
    tests/test_w84_constrained_policy_optimisation.py \
    tests/test_w84_streaming_substrate.py \
    tests/test_w84_tool_call_substrate.py
```

Expected: ~83 tests green in ~100 s.

## 5. The full file inventory

### Substrate / closure modules under `coordpy/`

* `transformers_runtime_v1.py` — W80/W86 HF transformers
  runtime with `precision_tier` + `device` + `skinny_trace`
  kwargs; the load-bearing runtime for #25/#26/#27/#30.
* `live_composed_memory_training_v1.py` — W86 #26 closure:
  live-vs-synthetic training of the W83 composed-learned-
  memory on real hidden states with projection + held-out
  disjointness.
* `long_context_intercept_bench_v1.py` — W86 #27 hidden-
  state-intercept-at-32k axis (skinny-trace + SDPA on Llama).
* `humaneval_real_bench_v1.py` — W86 #28: HumanEval bench
  with subprocess Python executor as critic + 3-arm head-to-
  head + content-addressed Merkle audit chain.
* `multi_host_distributed_substrate_v1.py` — W86 #29: gateway
  + partition-proxy entrypoints as container ENTRYPOINT +
  `run_multi_host_distributed_bench_v1` driver.
* `precision_tier_contract_v1.py` — W84 #30 (precision tier
  axis on the W80 contract).
* `streaming_substrate_intercept_v1.py` — W84 #32 (forward_
  stream + SSE + mid-stream hidden-state injection).
* `tool_call_substrate_v1.py` — W84 #33 (ToolCallSchemaV1 +
  ToolResultSchemaV1 + sandbox + 5-agent bench).
* `constrained_policy_optimisation_v1.py` — W84 #34
  (LagrangianRefinementV1 + projection helper).
* `lagrangian_with_projection_v1.py` — W86 #34 closure
  (Lagrangian + projection-fallback combined).
* `capacity_bench_harness_v1.py` — W84 #36 (3-axis curves +
  V1 indexed cache).
* `capacity_remediation_v2.py` — W86 #36 closure (deferred-
  graph-cid V2 cache; cliff moves 112×).
* `budget_enforcement_v1.py` — W84 #37 (RunBudgetSpecV1 +
  BudgetEnforcerV1 + CostModelV1).
* `budget_enforced_composed_recovery_v1.py` — W86 #37
  closure (composed-pipeline integration).
* `moe_runtime_substrate_v1.py` — W86 #31 (3 new W80 axes +
  ExpertRoutingSnapshotV1 + MoERuntimeAdapterV1 + closure
  bench).

### Drivers + verifiers under `scripts/`

* `run_frontier_closure_w86.py` + `verify_w86_audit_chain.py`
  — #25 / #26 / #27 driver + verifier.
* `colab_frontier_closure_w86.ipynb` — Colab notebook
  wrapping the frontier-closure driver in two process-
  isolated phases.
* `run_w86_humaneval_bench.py` + `verify_w86_humaneval_audit_chain.py`
  — #28 driver + verifier.
* `run_w86_multi_host_bench.py` + `verify_w86_multi_host_audit_chain.py`
  — #29 driver + verifier (one-shot docker-compose
  orchestrator).
* `run_w86_moe_substrate_closure.py` + `verify_w86_moe_audit_chain.py`
  — #31 driver + verifier (NEW).
* `colab_moe_substrate_closure_w86.ipynb` — Colab notebook
  for #31 (NEW).

### Infra under `docker/`

* `Dockerfile.coordpy-substrate` — python:3.12-slim +
  numpy + coordpy source; ENTRYPOINT runs the multi-host CLI.
* `compose-w86-multi-host.yml` — three services on bridge
  network with healthchecks and HMAC keys via env file.

### Tests under `tests/`

* `test_w86_*.py` — 50+ CI tests covering every W86 closure's
  surface (audit-chain re-derivation, skinny-trace API,
  precision-tier constants, MoE axes, etc.). Skip cleanly
  without torch/transformers/docker/NIM.

### Docs under `docs/`

* `RESULTS_W86_FRONTIER_CLOSURE.md` — #25/#26/#27 results.
* `RESULTS_W86_HUMANEVAL_HEAD_TO_HEAD.md` — #28 results.
* `RESULTS_W86_REAL_DISTRIBUTED.md` — #29 results.
* `RESULTS_W86_P1_CLOSURES.md` — P1 sweep results.
* `COLAB_PRO_RUNBOOK.md` — Colab Pro browser runbook.
* `W86_ISSUE_CLOSURE_COMMENTS.md` — DoD-mapped issue comments
  posted on GitHub for #25–#29 + #49.
* `W86_AUTOMATION_ARCHITECTURE.md` — this file.
* `THEOREM_REGISTRY.md` — proved-conditional + empirical
  claim ledger.
* `AUDIT_POST_W83_BLOCKERS.md` — per-issue verdict ledger.
* `HOW_NOT_TO_OVERSTATE.md` — do-not-overstate rules per
  milestone.

## 6. How the orchestration scripts work

### #29 multi-host orchestrator (`scripts/run_w86_multi_host_bench.py`)

```text
1. mint fresh HMAC keys for {alpha, beta, client}
2. write docker/.env.w86 with the keys
3. docker compose -p coordpy_w86 -f \
   docker/compose-w86-multi-host.yml --env-file ... up -d --build
4. wait for /healthz on host-a, host-b, partition-proxy
5. docker inspect → record hostnames + IPs + network_id into
   the MultiHostTopologyV1 capsule
6. build a client TrustRootV1 with the same HMAC key the
   containers know
7. run_multi_host_distributed_bench_v1(host_a_url, host_b_url,
   proxy_url, topology, client_trust_root, …) — exercises
   every #29 DoD bar against the live topology
8. write content-addressed report to disk
9. ALWAYS: docker logs tail → docker compose down -v →
   delete .env.w86
```

The bench exercises mTLS-shaped HMAC auth, cross-host
post-root match, partition test (1.5 s drop window via the
proxy's `/admin/start_drop` admin endpoint), skew test
(±2 s injected per container), 10× idempotent replay, and
RTT measurement — all in ~2 seconds wall.

### Frontier closure driver (`scripts/run_frontier_closure_w86.py`)

```text
--phase support:
  --skip-27         → Phase A: load full-trace runtime, run
                      #25 + #26, exit (OS reclaims VRAM)
  --skip-25 --skip-26 → Phase B: load skinny-trace runtime,
                        run #27

Multi-phase merge: if frontier_closure_report.json already
exists in --out-dir from a prior phase, the driver merges the
previous closures into the in-memory report so the final
write carries closure_25 + closure_26 (from Phase A) +
closure_27 (from Phase B) under one top-level report_cid.

Aggressive teardown between phases:
  - null runtime._model / _tokenizer / _torch
  - gc.collect()
  - torch.cuda.empty_cache()
  - torch.cuda.synchronize()
  - print VRAM free / total (diagnostic)
```

This is what makes the same notebook work on a 24 GB L4 AND
a 40 GB A100.

### HumanEval driver (`scripts/run_w86_humaneval_bench.py`)

```text
- corpus: SHA-256-verified against upstream
  openai/human-eval@312c5e5532f0e0470bf47f77a6243e02a61da530
- NIM Llama-3.1-8B-Instruct on every arm (anti-cheat: no model
  swap)
- 3 seeds × 30 problems × 3 arms × ~11 calls = 990 NIM calls
- per-call sidecar JSONL with full prompt+response (so a third
  party can re-verify CIDs without re-calling NIM)
- 1687 s wall on the W86 run
```

Three arms:
- A0: stock single-shot CoT @ t=0.0
- A1: K=5 samples @ t=0.7 + first-pass-among-K via subprocess
  executor (fair same-budget baseline)
- B:  solver_1 + solver_2 + critic (sees executor stderr) +
  reviser + judge — the multi-agent + executor-as-critic
  CoordPy path

### MoE driver (`scripts/run_w86_moe_substrate_closure.py`) — NEW

```text
1. import coordpy.moe_runtime_substrate_v1
2. probe_moe_capability_v1(model_name) — read config without
   loading weights; refuse if not MoE
3. MoERuntimeAdapterV1(model_name=…, device=cuda:0,
                       precision_tier=tier_bf16)
   - auto-detects MoE block class (Mixtral / OLMoE / Qwen-
     MoE / DeepSeek)
4. forward + per-block router-output hook → capture per-layer
   (expert_ids, gate_weights)
5. replay-from-KV; compare logits at bf16 tier
6. hidden-state intercept on a MoE block → verify CID moves
7. second forward → verify routing is deterministic (the
   "routing is state" claim)
8. write content-addressed MoESubstrateClosureBenchReportV1
9. exit 0 iff every load-bearing bool is True
```

## 7. The Colab Pro browser workflow (what you do)

This is the single user-driven step in the entire stack.

### For the FIRST run

1. Mint a HuggingFace token at
   https://huggingface.co/settings/tokens (role: Read; name:
   `coordpy-w86`).
2. Accept the Meta Llama-3.1 community license at
   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   (only needed for #25/#26/#27 — OLMoE for #31 is not
   gated).
3. Open the notebook URL (e.g.
   https://colab.research.google.com/github/adotdong29/CoordPy/blob/main/scripts/colab_moe_substrate_closure_w86.ipynb).
4. **Runtime → Change runtime type → A100 GPU** (or L4 / V100;
   T4 is borderline for the frontier-closure notebook).
5. Click the 🔑 key icon in the left sidebar →
   *+ Add new secret* → name = `hf_token`,
   value = `hf_xxxxxxxx`. Toggle *Notebook access* **on**.

### For every subsequent run

1. Open the notebook URL fresh (Colab fetches the latest from
   GitHub each time).
2. *Runtime → Disconnect and delete runtime* (clean start —
   important so `/content` is empty).
3. *Runtime → Run all*.
4. **At cell 8** (the Drive-save cell) Colab will prompt for
   Drive authorization — say yes. Results save to
   `MyDrive/coordpy_*/w86_*/`.
5. The notebook also offers a zip download at the end (backup
   in case Drive sync fails).

### Sharing results back

Either drop the zip in chat, or paste the contents of
`*_report.json` directly. The audit chain is tiny (~10 KB for
the MoE report, ~150 KB for the HumanEval report) — easy to
paste.

### Cost

Zero — Colab Pro is a flat monthly subscription, NOT per-run
billing.

## 8. The honesty contract

Every closure in W86 ships with explicit carry-forward
limitations recorded in `docs/THEOREM_REGISTRY.md` (with
`W86-L-*` IDs) and `docs/HOW_NOT_TO_OVERSTATE.md` (with
per-milestone do-not-overstate rules). These are mandatory
reading before any W86 claim is repeated:

* `W86-L-LLAMA-3.1-8B-WRITE-ATTENTION-BIAS-GQA-CAP` — Llama's
  GQA breaks the W80 attention-mask hook.
* `W86-L-CONFORMANCE-SUITE-NOT-PRECISION-TIER-AWARE-CAP` —
  conformance harness hardcodes fp32 floor; the runtime
  correctly reports tier-correct byte-id.
* `W86-L-FRONTIER-RUNTIME-V1-COLAB-PRO-CAP` — the W86 closure
  ran on Colab Pro; runtime-agnostic, re-runnable anywhere.
* `W86-L-LIVE-CM-TRAIN-V1-PROJECTION-CAP` — #26 live training
  uses an R^4096 → R^8 projection to make the live task
  comparable to the W83 synthetic baseline.
* `W86-L-LIVE-CM-TRAIN-V1-NEXT-STEP-TASK-CAP` — V1 trains on
  next-step hidden-state forecasting; richer tasks are V2.
* `W86-L-LONG-CONTEXT-INTERCEPT-V1-SKINNY-TRACE-CAP` — #27 at
  32 k uses skinny-trace + SDPA; full per-layer hidden-state
  capture at 32 k is V2.
* `W86-L-PRECISION-TIER-BF16-WIDENED-FLOOR-CAP` — replay-
  from-KV at bf16 carries tolerance 0.5 vs fp32's 5e-3.
* `W86-L-HUMANEVAL-V1-A1-SAME-BUDGET-NOT-BEATEN` — B beats
  A0 (stock) on #28 by +7.8 pp but does NOT beat the harder
  same-budget A1 baseline (visible-test filter K=5).
* `W86-L-HUMANEVAL-V1-SUBPROCESS-PYTHON-EXECUTOR-CAP` —
  CPython subprocess sandbox; out-of-process side effects
  not blocked.
* `W86-L-MULTI-HOST-DISTRIBUTED-V1-DOCKER-BRIDGE-CAP` — #29
  closure is on docker-compose bridge network, NOT real
  multi-physical-machine WAN.
* `W86-L-MULTI-HOST-DISTRIBUTED-V1-HMAC-NOT-X509-CAP` — mTLS
  shape is HMAC-SHA256, not X.509 TLS.
* `W86-L-QUANT-INT8-NEEDS-BNB-CUDA-COLAB-CAP` — #30 int8
  bullet still requires bitsandbytes + CUDA.
* `W86-L-LAGRANGIAN-V1-FLOOR-NOT-RESPECTED-ALONE-CAP` — #34
  closure is via Lagrangian + projection, not Lagrangian-
  only.
* `W86-L-MOE-SUBSTRATE-V1-NEEDS-CUDA-AND-MOE-WEIGHTS-CAP` —
  #31 closure (pending) requires CUDA + real MoE weights.
* `W86-L-MOE-SUBSTRATE-V1-TOPK-RESTORE-CAP` — V1 restores
  top-K expert IDs + gate weights, not the full router-
  logits distribution.
* `W86-L-MOE-SUBSTRATE-V1-HF-FAMILIES-CAP` — V1 supports
  Mixtral / OLMoE / Qwen-MoE / DeepSeek; custom routers V2.

## 9. Stable boundary

Every W86 closure preserves the stable SDK release contract:

* `coordpy.__version__ == "0.5.20"` byte-for-byte unchanged.
* `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"` byte-for-byte
  unchanged.
* `coordpy/__init__.py` byte-for-byte unchanged.
* No PyPI publish.
* All W86 modules are explicit-import only (no re-exports via
  `coordpy.__experimental__`).

## 10. Where things go next

The single remaining open issue is **#31 (MoE substrate)** —
the code is shipped (`coordpy/moe_runtime_substrate_v1.py`),
the driver + verifier + Colab notebook are ready, the CI
surface tests pass. The empirical closure waits on one Colab
Pro A100 run with the OLMoE-1B-7B model.

After that lands, meta-#49's complete post-W83 P0 + P1 line
is closed. Subsequent work would belong in fresh issues:

* P2 line (#38–#45): Byzantine fault tolerance, differential
  privacy, MPC, schema evolution, drift detection, multi-
  tenancy, GPU/TPU determinism, event-graph GC.
* P3 line (#46–#48): multi-modal substrate, observability,
  formal verification.
