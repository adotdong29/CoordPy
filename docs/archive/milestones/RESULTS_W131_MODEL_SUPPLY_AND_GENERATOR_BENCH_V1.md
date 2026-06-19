# RESULTS W131 — code-competent MODEL supply census + stronger-generator hard-cluster dev bench (W129 selector FIXED) + stronger-model gate

Executes the pre-registered `docs/RUNBOOK_W131.md` α/β/γ branch logic (LOCKED before any hosted-NIM
dev-bench call). All numbers below are filled ONLY from emitted verdict JSON
(`feedback_never_prewrite_results_before_data`).

## TL;DR

W130 found the hard-cluster ceiling generation-bound and the dominant failures capability-bound, so
W131 attacked the **MODEL-capability axis**: census the reachable code-competent model supply, then
swap a stronger generator into the SAME 11 hard-cluster EXPOSED dev bench at MATCHED K=5 with the
W129 NIM-free SOLEAD selector **held FIXED downstream** — GENERATION (the base model) the only
variable.

* **Lane α (census, $0).** The supply landscape CHANGED since W124, but the binding gap MOVED rather
  than closed. Local-HF transformer runtime is **DEAD** (no `torch`/`transformers` under this
  interpreter). Local-Ollama and hosted-NIM are both **LIVE**: the census found **13 reachable
  stronger-than-Maverick code models** on the NIM catalogue (Qwen3-Coder-480B, DeepSeek-V4-pro,
  Qwen3.5-397B, Mistral-Large-3-675B, GLM-5.1, …) plus local code models. **But FRONTIER_ELIGIBLE =
  NONE** — every stronger model is UNKNOWN-from-primary on training cutoff ⇒ DEV_ONLY
  (resistant-ineligible). The supply gap is no longer "no strong code model exists/loads" (W124) — it
  is **"no PRIMARY-KNOWN-cutoff stronger model on the ICPC family"** (cutoff DISCLOSURE, not model
  existence).
* **Lane β (dev bench).** The strongest reachable code model, **Qwen3-Coder-480B** (a35b active,
  ~28× Maverick's active params), run PLAIN + GG2-rewrite at K=5 with the W129 selector fixed,
  produced **1 "new" pool solve = `doubleup`** (the SAME problem W130's GG2 cracked on Maverick) and
  **NOTHING genuinely new** ⇒ **NOT EARNED** (1 < the +2-spanning bar). The W130-winning rewrite
  method (B3_GG2) added **0** over plain on the 480B; the trio was preserved (0 new mis-commits).
  [Local-7B rung + 480B method-arm escalation: § 3.3–3.4.]
* **Lane γ (gate).** `NO_CERTIFIABLE_STRONGER_MODEL`, decision CID `258b6ed7` **invariant**
  (W114→W131), {KNOWN:1 (Maverick Aug-2024, SETTLED), UNKNOWN:4}. T1 [§ 4] ⇒ targeted resistant
  probe NOT earned ⇒ **$0 resistant NIM**.

**W89 (+5.56) + W105 (+7.00) STAND as the only two confirmed retirements.** [Cap registration § 5.]

---

## § 1 — Lane α: code-model supply + capability census ($0)

Module `coordpy.code_model_supply_census_v1`; driver `scripts/run_w131_model_supply_census_v1.py`;
verdict `results/w131/census/model_supply_census_v1.json` (`census_cid d360c117e54a65e4…`,
34 records, $0).

| surface | status | best code-competent candidate | usage |
|---|---|---|---|
| **LOCAL_HF** (`transformers_runtime_v1` / `code_substrate_v1`) | **DEAD** — `torch`/`transformers` not importable (ModuleNotFoundError) under Python 3.14.5 | — | NOT_A_GENERATOR |
| **LOCAL_OLLAMA** (`http://localhost:11434`, OpenAI-compatible) | LIVE, $0 | `qwen2.5-coder:32b` (best local) → throughput-capped, see § 3.3; `qwen2.5-coder:7b` (fast) | DEV_ONLY |
| **HOSTED_NIM** (`integrate.api.nvidia.com`, 118-model catalogue) | LIVE | `qwen/qwen3-coder-480b-a35b-instruct` (strongest reachable code model) | DEV_ONLY |

* **`FRONTIER_ELIGIBLE` = NONE.** 13 reachable **stronger-than-Maverick** code models, all
  UNKNOWN-from-primary on cutoff ⇒ **32 DEV_ONLY** records. Maverick is the only PRIMARY_KNOWN model
  and is **SETTLED**.
* **Code-competence rigor:** the smoke gate (`code_smoke_gate_v1`, 2-case constant-output guard) ran
  on the local code models — `qwen2.5-coder:32b` PASS, `qwen2.5-coder:7b` PASS, `lexi-coder:8b` PASS,
  `deepseek-r1:7b` FAIL (reasoning model, no clean code at 300 tokens). Hosted code smoke deferred to
  the dev-bench canary ($0 census).
* **Load-bearing parsing fix (`normalize_fence_v1`).** Qwen-Coder models emit the fence info-string
  on its own line (` ```\npython\n<code> `), which crashes the audited `extract_candidate_code_v1`
  (stray `python` line → `NameError` → RC:1) — discovered when the 32B smoke FAILED `<RC:1>` despite
  emitting correct logic. Fixed at the generation seam (parsing fairness only, NOT a capability
  lever; the call sidecar preserves the raw model output). Without it a stronger-model dev bench
  would be INVALID (it would penalize Qwen-Coder by a formatting quirk — the W128 sketch-parser-bug
  lesson).

---

## § 2 — Lane γ: stronger-model cutoff-disclosure gate ($0)

Module `coordpy.stronger_model_cutoff_certification_v1` (reused, C1∧C2∧C3∧C4); driver
`scripts/run_w131_stronger_model_gate_recheck_v1.py`; verdict
`results/w131/stronger_model_gate/gate_recheck_v1.json`.

* **Verdict `NO_CERTIFIABLE_STRONGER_MODEL`; decision CID `258b6ed794b45a18…` INVARIANT** (byte-
  identical W114→W131); {KNOWN:1 (Maverick, primary cutoff 2024-08-31), UNKNOWN:4 (Qwen3-Coder-480B
  / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5)}. $0 NIM.
* **W131 census cross-reference:** 13 reachable stronger-than-Maverick code models, **0
  FRONTIER_ELIGIBLE** — every one UNKNOWN-from-primary on cutoff. A model SUPERSEDES Maverick only if
  it becomes primary-KNOWN AND certifiable on the matched ICPC family; none does.
* **Frontier-eligible vs dev-only rule (§ 9 of the runbook).** FRONTIER_ELIGIBLE requires
  primary-KNOWN cutoff ≤ the resistant frontier (Maverick's Aug-2024; the W120 resistant slice is
  post-cutoff). The strong 2025-era models post-date that frontier even if a cutoff were disclosed,
  so a resistant probe with them would be a contaminated-benchmark-sold-as-frontier-win. Maverick
  (the lone reachable KNOWN-cutoff model) is settled.

---

## § 3 — Lane β: stronger-generator hard-cluster dev bench (W129 selector FIXED)

Module `coordpy.generator_model_bench_v1`; driver `scripts/run_w131_generator_model_bench_v1.py`.
Same 11 hard-cluster EXPOSED dev targets (`hard_dev_target_cid 546c146624b14d4e…`, byte-identical to
W128/W129/W130), MATCHED K=5, W129 NIM-free SOLEAD selector held FIXED downstream. Capability ladder:
Maverick-17B-128e (B0, reuse) → Qwen2.5-Coder (local, $0) → Qwen3-Coder-480B-a35b (hosted, the
strongest reachable). Old W128/W129 pool ceiling = 3/11 ({pawnshop, sunandmoon, blueberrywaffle});
plain baseline 2/11; W130's GG2 cracked `doubleup`.

### § 3.1 — B0: Maverick-17B baseline (reused from W130, $0)

Old pool ceiling 3/11; plain baseline 2/11; W130 GG2 rewrite cracked `doubleup` (the only W120–W130
generation crack). No re-run.

### § 3.2 — Qwen3-Coder-480B-a35b (HOSTED, the strongest reachable code model; 110 NIM core)

`results/w131/dev_bench/verdict_qwen_qwen3-coder-480b-a35b-instruct.json` (run
`…hosted480b_core2`; 110 NIM, K=5, 0 same-budget violations, 0 job-drops; the first attempt at
workers=8 hit HTTP-429 throttling and was re-run at workers=2 with 429-aware backoff).

| arm | new pool solves | committed | note |
|---|---|---|---|
| **B1_PLAIN** | **1** {`doubleup`} | 1 {`doubleup`} | adhoc_math / HIDDEN_EDGE; real + leakage-clean |
| **B3_GG2** (W130 rewrite) | **0** | 0 | added nothing over plain on the 480B |

**EARN GATE = NOT EARNED** (`GG_EXPOSED_DEV_BENCH_NOT_EARNED`): best arm B1_PLAIN created 1 new pool
solve (need ≥2 spanning ≥2 families/modes); `spans_two=False` (winners_real=True, winners_clean=True).

**The sharp finding:** the strongest reachable code model (≈28× Maverick's active params) PLAIN-solves
**only `doubleup`** — the SAME problem W130's GG2 rewrite already cracked on Maverick — and produces
**ZERO genuinely-new solves** beyond the W120–W130 pool. Model capability (17B→480B) does **not** move
the hard-cluster ceiling. Stored-regression trio PRESERVED (0 new mis-commits): `pawnshop` committed
**correct** via SINGLE_SURVIVOR (better than Maverick's W128 wrong-A0 mis-commit), `sunandmoon`
abstain-safe (both-correct tie), `blueberrywaffle` the 480B did not generate a passing candidate
(generation regression vs Maverick's pool) but the selector correctly did not mis-commit.

### § 3.3 — Qwen2.5-Coder local rung ($0)

The local lane is reachable + code-competent but **throughput/reliability-capped on this host**, so
it could not contribute a completed bench rung — registered honestly, not hidden:

* **`qwen2.5-coder:32b`** — reachable + code-competent (smoke PASS) but runs **CPU-bound**
  (`size_vram=0`, ≈1 token/s; Metal does not engage for the 32B Q4 on this host), so a 1536-token
  solution takes ≈25 min PER CALL and a 110-call bench would take days. A hardware throughput cap.
* **`qwen2.5-coder:7b`** (the fast Metal fallback) — solves the smoke task in ≈10 s clean, but under
  this session's sustained load (the hosted 480B run + repeated 32B load/unload/kill cycles) the
  Ollama daemon degraded: the temp-0.0 plain canary looped to the 1536-token cap and timed out, and
  even with `--no-canary` the temp-0.2 jobs stalled (>300 s) — across repeated attempts the 7B did
  not complete a clean 11-target run. Registered as a local-lane reliability cap on this loaded host.

**Net for the local lane:** it cannot practically supply a stronger-than-Maverick generator — the
strong local coder (32B) is CPU-bound and the fast one (7B) is both weaker than Maverick and
unreliable here. This **reinforces** `W131-T-FRONTIER-ELIGIBLE-SUPPLY-IS-CUTOFF-DISCLOSURE-BOUND`:
the only practical stronger-than-Maverick supply is the hosted DEV_ONLY catalogue. The model-axis
verdict therefore rests on the **decisive hosted Qwen3-Coder-480B** rung (§ 3.2 + § 3.4), which
completed cleanly (220 NIM, 0 same-budget violations, 0 job-drops) and is the strongest reachable
code model — strictly stronger than any local candidate.

### § 3.4 — 480B method-arm escalation (B2_RDIV, B4_GGLEAD)

Per RUNBOOK § 5, the hosted core showed ≥1 new pool solve, so the method arms were run on the 480B to
make the full-slate-on-strongest-model result airtight (`…hosted480b_escalate`; 110 NIM, 0
same-budget violations, 0 job-drops).

| arm | new pool solves | committed | note |
|---|---|---|---|
| **B2_RDIV** (role-diverse / GG1) | **0** | 0 | reached a correct `blueberrywaffle` in-pool; selector abstained (under-determined) |
| **B4_GGLEAD** (GG1→GG2 composite) | **0** | 0 | reached a correct `sunandmoon` in-pool; selector abstained |

**EARN GATE = NOT EARNED.** So the FULL slate on the strongest reachable code model is:
**B1_PLAIN = 1 (`doubleup`, the W130 crack) · B2_RDIV = 0 · B3_GG2 = 0 · B4_GGLEAD = 0** ⇒ **0
genuinely-new solves beyond the W120–W130 pool.** The method arms (role-diverse, counterexample-
rewrite, composite) add **nothing** on the 480B — exactly mirroring W130's finding on Maverick (the
methods added 0; only one arm reached the W130 `doubleup`). Trio PRESERVED (0 new mis-commits): the
role-diverse / composite arms can GENERATE correct versions of the pool-bearing problems
(`blueberrywaffle`, `sunandmoon`) but the fixed W129 selector correctly ABSTAINS on the
under-determined ties — the W128/W129 selection discipline holds unchanged under a stronger
generator.

---

## § 4 — Targeted resistant probe (T1 ∧ T2)

* **T1 (a W131 arm earns genuinely-new headroom on the EXPOSED dev bench) = FALSE.** The strongest
  reachable code model (Qwen3-Coder-480B), full slate B1–B4, W129 selector fixed, K=5, created **0
  genuinely-new** solves (only `doubleup`, the W130 crack) ⇒ NOT EARNED (1 < +2). The local 7B floor
  rung (§ 3.3, weaker than Maverick) cannot make T1 true.
* **T2 (the earning model is FRONTIER_ELIGIBLE, or the method translates onto a FRONTIER_ELIGIBLE
  target) = MOOT** (T1 false), and would FAIL anyway: FRONTIER_ELIGIBLE = NONE (§ 2), and the
  contamination/memorization rule (RUNBOOK § 8) makes any EXPOSED earn by a DEV_ONLY model
  resistant-ineligible.
* ⇒ **targeted resistant probe NOT earned ⇒ $0 resistant NIM.** Exposed frontier-control NOT bought
  (resistant-first). No n=30 seed-chasing; no 405B; no reopening MBPP+ V2 / frozen cross-modal /
  Llama-3.1 rescue / APPS main-lane.

---

## § 5 — Carry-forward registration

* **W89 (+5.56) + W105 (+7.00) STAND as the only two confirmed retirements.** W131 retires none and
  adds none.
* **`W131-L-MODEL-AXIS-GENERATION-CEILING-DEV-BENCH-CAP`** (empirical) — a stronger code model (the
  hosted Qwen3-Coder-480B full B1–B4 slate; local rungs § 3.3), W129 selector held FIXED, K=5,
  creates 0 genuinely-new EXPOSED hard-cluster solves ⇒ the generation ceiling is **not
  model-axis-liftable at the reachable rungs**. The MODEL-axis sibling of the W123→W130 cap taxonomy
  (battlefield → encoder → re-routing → synthesis → scaffold-gen → role-diverse-search →
  selection-oracle → generator-line → **model-axis**).
* **`W131-T-STRONGER-MODEL-DOES-NOT-CRACK-NEW-HARD-CLUSTER-PROBLEMS`** (empirical) — a ~28× larger
  frontier code model PLAIN-reproduces exactly the W130 `doubleup` crack and nothing else; the
  dominant `WRONG_ALGORITHM_*` failures are unmoved ⇒ they are capability failures the reachable
  model frontier does not close, reinforcing `W130-T-ADMISSIBLE-SKETCH-IS-CAPABILITY-NOT-GENERATION-
  FIXABLE`.
* **`W131-T-FRONTIER-ELIGIBLE-SUPPLY-IS-CUTOFF-DISCLOSURE-BOUND`** (census/gate) — 13 reachable
  stronger-than-Maverick code models, ALL UNKNOWN-from-primary on cutoff ⇒ DEV_ONLY;
  FRONTIER_ELIGIBLE = NONE. The model-axis supply gap MOVED from W124's "no strong code model
  exists/loads" to "no PRIMARY-KNOWN-cutoff stronger model on the ICPC family" (disclosure, not
  existence).
* Carried forward unchanged: `W128-L-GRAPH-FLOW-EXPOSED-SUPPLY-CAP`,
  `W129-L-HARD-CLUSTER-GENERATION-CEILING-CAPS-SELECTION-EARN`,
  `W130-L-GENERATION-CEILING-DEV-BENCH-CAP`; stronger-model gate `258b6ed7` invariant.

---

## § 6 — Spend + artifacts

* **$0** on Lane α census (local smoke + reachability) and Lane γ gate.
* Lane β NIM: hosted Qwen3-Coder-480B core = 110 NIM (+ ~31 wasted on the 429-throttled first
  attempt); escalation [§ 3.4 pending]; local 7B = $0. No resistant NIM.
* 2 new explicit-import-only modules (`code_model_supply_census_v1`, `generator_model_bench_v1`) +
  3 scripts + `tests/test_w131_model_supply_census_and_bench_v1.py` (17 tests). No version bump
  (0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.
