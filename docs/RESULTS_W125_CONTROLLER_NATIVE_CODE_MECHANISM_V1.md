# RESULTS — W125: hosted controller-native code mechanism on resistant ICPC (3 lanes)

**Date:** 2026-05-31 · **Lane α verdict:** controller-native mechanism is **REAL** (not fake-different) + **contract-clean** (lead `C3`, cid `7989655f…`) · **Lane β verdict:** `FRESH_RESISTANT_PILOT_NOT_EARNED_HEADROOM_CAP` (`blind_selection_headroom = 0`)
**Spend:** **$0 NIM** (no fresh pilot earned). Decision CID `258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched. `ultracode` OFF throughout.

## Why W125 mined the hosted controller stack

W120–W124 closed the two cheap levers: the **battlefield** (resistant +0.00 FAIL / exposed +3.33 FAIL / 3-seed B4 unresolvable at n=30 / ≥100-per-field supply-UNREACHABLE) and the **local encoder** (distilgpt2 hidden state adds nothing; no code-competent local model). W125 mines the **third, unused lever** — the hosted/controller arsenal already in the repo — and promotes the W124 M6 *contract-only* PATCH/REPLAN/ABSTAIN controller into a real, executable, genuinely controller-native mechanism. The question is **not** "can a different prompt do slightly better?" but: **can our own hosted controller stack beat same-budget self-consistency (A1) on the official resistant ICPC family where reflexion (B) failed?** Resistant-first; exposed control only if the resistant line earns it. Pre-registered in `docs/RUNBOOK_W125.md`, locked BEFORE the decisive probe.

## Lane α — controller-native mechanism (MAIN, NIM-free): REAL + contract-clean

`coordpy/controller_native_code_mechanism_v1.py` (explicit-import-only) is the **first repo module** to wire the hosted-controller stack (`hosted_cache_aware_planner_v12` shared-prefix+per-role plan; `hosted_logprob_router_v12` abstain floor; `hosted_router_controller_v12` routing schema) to the **audited** `tool_call_substrate_v1` plane and the `executor_grounded_patcher_v1` typed failure digest, on the official-ICPC code path (`icpc_reflexion_bench_v1` grader). graphify confirmed the hosted router and the tool-call substrate were 6 hops apart with **no semantic edge** — this module creates the bridge.

**The slate (RUNBOOK § 2):** **C1** role-specialized planner/controller; **C2** router-selected multi-candidate controller; **C3** tool-substrate audited repair loop (digest-routed PATCH/REPLAN/ABSTAIN) — the LEAD.

**Structural fake-different test (RUNBOOK § 4) — it BITES.** Over {reflexion B, C0 (negative control: reflexion relabeled), C1, C2, C3}:

| candidate | distinct actions | audited tool plane | digest-routed retry | linear chain | classification |
|---|---|---|---|---|---|
| reflexion B | 1 | no | no | yes | **FAKE_DIFFERENT** |
| C0 (control) | 1 | no | no | yes | **FAKE_DIFFERENT** |
| C1 role planner | 3 | yes | yes | no | REAL |
| C2 router-select | 2 | yes | no | no | REAL |
| **C3 tool-substrate repair** | **4** | **yes** | **yes** | **no** | **REAL (lead)** |

Reflexion B and the deliberately-degenerate C0 both classify `FAKE_DIFFERENT` (the test discriminates); C1/C2/C3 are genuinely controller-native; the lead is **C3** (the audited-tool-plane + digest-router superset of the W124 M6 contract and the W111 M3 patcher lineage).

**Four NIM-free contract checks (RUNBOOK § 5) — all PASS** (on the controlled synthetic substrate; re-verified clean on the real resistant problem too): (1) tool-call **audit chain re-hashes** byte-identically + a tampered result byte flips the Merkle root + idempotent re-commit refused; (2) **grader-call capture** complete + **never-reads-secret** (the controller's in-loop feedback is public-sample-only; the secret grade returns only a boolean; a deliberate leak is caught); (3) **routing determinism** (same pool → identical plan/route/outcome CIDs); (4) **same-budget accounting** (model-generation budget ≤ K=5, exactly one final secret grade). **Lane α conclusion: the hosted controller stack is a real new mechanism on code, not fake-different.**

## Lane β — resistant-first $0 headroom replay: NOT earned (generation-capped)

A **$0 replay** over the 330 already-paid real Maverick generations on the EXACT W120 resistant 30-slice (CID `01bf9ef8…`), re-grading every generation on the official secret + sample cases (1 seed × 30 problems × 11 generations; ~22 min; **$0 NIM**):

| quantity | value | reading |
|---|---|---|
| A1 pass@5 (secret) | **7 / 30** | reproduces W120 A1 = 23.33 % exactly |
| **pool-union** (A0∪A1∪B on secret) | **8 / 30** | the ENTIRE real Maverick generation pool reaches only 8/30 |
| oracle_pool_headroom (union − A1) | **+1** | the absolute ceiling of ANY pool re-routing — and it needs an oracle, and +1/30 = +3.33pp is inside the ±3.34pp null band |
| C2 blind-select committed (secret) | **7 / 30** | selection over the A1 pool can't beat pass@5 |
| C3 digest-routed walk committed (secret) | **7 / 30** | the lead controller also commits exactly A1 |
| **`blind_selection_headroom`** | **0** | ZERO A1-fail problems where a hidden-test-blind controller commits a secret pass |
| `reflexion_divergence` | **23 / 30** | reflexion gets STUCK (repeated candidate/digest) on 23 problems |
| `looks_right_fails_hidden` | **10** | 10 generations pass ALL public samples but FAIL secret |

**The two killer diagnostics, together, explain why no controller can win here at $0:** reflexion is stuck on 23/30 (a digest-router *would* diverge), **but there is nothing to route to** — the pool caps at 8/30 (the 22 hard problems are uniformly unsolved across all 11 generations), and the only hidden-test-blind signal a controller has, the public sample tests, is **non-discriminating** (10 candidates pass all samples yet fail the hidden cases). C2 selection and C3's digest-routed walk therefore both commit exactly A1's 7/30; `blind_selection_headroom = 0`.

**Pilot earn gate (RUNBOOK § 6):** E1 (contract clean) ✓ ∧ E2 (lead is REAL) ✓ ∧ **E3a (`blind_selection_headroom ≥ 2`) ✗ (= 0)** ∧ E3b (`reflexion_divergence ≥ 3`) ✓ ⇒ **`FRESH_RESISTANT_PILOT_NOT_EARNED_HEADROOM_CAP`**. The resistant field is **generation-capped** for $0 controller re-routing; the controller's potential value lives entirely in NEW trajectories on the 22 uniformly-unsolved problems, which the $0 corpus cannot supply and for which there is **no precursor signal** to fund a hosted spend. Reflexion (the closest measurable mechanism) already gave +0.00. **$0 NIM, no fresh pilot.**

## Lane β earn / no-earn — exposed control

Per RUNBOOK § 8: because the resistant pilot is **NOT earned**, the matched exposed ICPC control is **NOT earned and NOT bought**. Resistant-first is the frontier move; the exposed control is worth buying only when the new mechanism gives something real to interpret on the resistant field — it did not.

## Lane γ — stronger-model gate / truth

`coordpy.stronger_model_cutoff_certification_v1` re-affirmed **`NO_CERTIFIABLE_STRONGER_MODEL`**, decision CID **`258b6ed794b45a18…` invariant**, registry `{KNOWN:1, UNKNOWN:4}` (Maverick KNOWN Aug-2024 certifiable-but-settled; Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5 UNKNOWN-from-primary). Maverick remains the only certifiable hosted target. The W125 spend gate is **Lane β (resistant headroom), not Lane γ**; the local transformer-native line stays CLOSED (no new local code-model supply).

## Carry-forward (unchanged retirements)

Exactly **TWO** confirmed retirements stand — **W89** (base HumanEval × llama-3.3-70b, +5.56 pp) and **W105** (HumanEval+ × llama-3.3-70b, +7.00 pp), both contamination-EXPOSED HumanEval-family at 70B. W125 **retires none and adds none**. It registers a **limitation** carry-forward, `W125-L-RESISTANT-GENERATION-CAP`: the resistant ICPC field is generation-capped for $0 controller re-routing (pool-union 8/30, A1 7, blind headroom 0, public-sample signal non-discriminating) — the **mechanism-lever** sibling of W123's post-cutoff **battlefield-supply** cap and W124's **local-encoder-supply** cap. The honest result is a sharp, executable **negative on the spend question** alongside a **positive on the mechanism question** (the controller arsenal is genuinely controller-native and contract-clean) — the arsenal was mined hard, not waved away as "battlefield capped".

## Named claims (THEOREM_REGISTRY)

- `W125-T-CONTROLLER-NATIVE-MECHANISM-REAL-NOT-FAKE-DIFFERENT` (mechanically-checked): the composed hosted-controller/tool-substrate/digest mechanism is genuinely controller-native (≥2 native properties + non-linear control flow) vs reflexion's linear DRAFT chain; reflexion B + the C0 control classify FAKE_DIFFERENT, proving the test discriminates; all four NIM-free contract checks pass.
- `W125-L-RESISTANT-GENERATION-CAP` (empirical): on the W120 resistant 30-slice the real Maverick generation pool reaches 8/30, A1 captures 7, and a hidden-test-blind controller's selection headroom is 0 (oracle ceiling +1, within the null band); a $0 controller re-routing cannot move B−A1, so a fresh hosted pilot is not precursor-earned.
- `W125-T-RESISTANT-PUBLIC-SAMPLE-SIGNAL-NON-DISCRIMINATING` (empirical): 10 resistant generations pass ALL public samples but fail secret ⇒ the only hidden-test-blind in-loop signal a controller has is non-discriminating on this field.

## Artifacts

- `coordpy/controller_native_code_mechanism_v1.py` (explicit-import-only; C0/C1/C2/C3 + audited grader plane + blind scorer + structural fingerprint + 4 contract checks + headroom probe + earn gate).
- `scripts/run_w125_lane_alpha_contract_checks_v1.py`, `scripts/run_w125_lane_beta_resistant_headroom_replay_v1.py`, `scripts/run_w125_stronger_model_gate_recheck_v1.py`.
- `tests/test_w125_controller_native_code_mechanism_v1.py` (13 tests; falsifiability-first; validated by direct execution — local pytest/attrs env broken).
- `results/w125/lane_alpha/{slate_verdict.json,contract_verdict.json}`, `results/w125/lane_beta/headroom_verdict.json`, `results/w125/stronger_model_gate/gate_recheck_v1.json`.
- `docs/RUNBOOK_W125.md` (pre-registration, locked before the probe).
