# RUNBOOK W125 — Hosted controller-native code mechanism on the official ICPC family (resistant-first; exposed control only if earned)

**Status: PRE-COMMITTED. Locked BEFORE any decisive probe and BEFORE any NIM call.**
Fill `docs/RESULTS_W125_*` ONLY from the emitted verdict JSON (the "never pre-write
results" discipline — `feedback_never_prewrite_results_before_data`). The pre-committed
code rule below is the branch authority, not any prior.

`ultracode` is OFF for W125. W125 is a bounded controller-mechanism / hosted-pilot
milestone, not a repo-wide dynamic-workflow job. It turns ON only if the work
unexpectedly expands into a genuine dynamic-workflow problem (multiple hosted controller
candidates all earn live runs at once / repo-wide controller-substrate migration / broad
multi-surface external verification at once) — and only after an explicit mode-change note.

Stable boundary (unchanged, asserted in tests):
`coordpy.__version__ == "0.5.20"`, `coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`,
no PyPI publish, `coordpy/__init__.py` untouched, advanced work explicit-import only.

---

## § 0 — Why W125 is NOT more battlefield and NOT another encoder probe

W120–W124 closed the two cheap levers:

* **Battlefield lever (W120–W123).** Resistant ICPC Maverick pilot FAIL (B−A1 **+0.00pp**,
  MLB-2 8%); matched exposed control FAIL (**+3.33pp**, in the null band); n=30 paired
  contrast unresolvable (terminal B4: resistant 3-seed mean **+4.44**, exposed **+8.89**);
  the ≥100-per-field move is **supply-capped** (resistant hard-capped ≈45 tier-1 on the
  4 official post-cutoff package surfaces).
* **Local-encoder lever (W124).** The transformer-native hidden-state read on the only
  loadable local model (distilgpt2, 82M general LM) gave **no gain** (AUC_h 0.6345 ≈
  AUC_s 0.6343); blocked at the precursor on **local code-model-encoder supply**. M5
  gated off; the M6 deterministic tool-call-substrate controller shipped **contract-only**
  for lack of a competent local generator.

W125 mines the **third, unused lever**: the hosted/controller arsenal already in the repo
(`hosted_router_controller_v12`, `hosted_logprob_router_v12`,
`hosted_cache_aware_planner_v12`, `multi_agent_substrate_coordinator_v15`,
`tool_call_substrate_v1`, `executor_grounded_patcher_v1`). The question is NOT "can a
different prompt do slightly better?" It is: **can our own hosted controller stack beat
same-budget self-consistency (A1) on the official resistant ICPC code family where
reflexion (B) failed?** W125 promotes the W124 M6 contract into a real, executable,
genuinely controller-native mechanism, tests it on the resistant field FIRST, and buys
the matched exposed control ONLY if the resistant line earns interpretive value.

W125 is NOT a bounded-context / compaction / summarization / "cram less / truncate better"
trick (those remain anti-patterns, not the frontier path).

---

## § 1 — α / β / γ branch logic (pre-committed)

* **Lane α (mechanism, MAIN, NIM-free).** Build the controller-native slate C1/C2/C3
  (§ 2) on the existing official ICPC code path. Run the four NIM-free contract checks
  (§ 5) + the structural "fake-different vs real" test (§ 4). Choose the lead candidate
  by dry-run structural dominance; kill the rest honestly. Lane α answers: *does the
  hosted controller stack create a real new mechanism on code, or is it fake-different
  and not worth Maverick spend?*

* **Lane β (resistant-first, $0 precursor → conditional spend).** Run the lead controller
  as a **$0 replay** over the already-paid real Maverick generation corpus on the **W120
  resistant** field (§ 3) and measure resistant **headroom** (§ 6). Apply the pilot earn
  gate (§ 6). If EARNED → the cheapest honest fresh resistant pilot is admissible (§ 7).
  If NOT earned → **$0 NIM, no fresh pilot**, ship the contract-verified mechanism + register
  the cap. Exposed control is downstream of resistant (§ 8).

* **Lane γ (stronger-model gate / graphify / truth).** Re-check primary-source cutoff
  disclosures (§ 9). graphify refreshed START + END (§ 11). Tighten the truth surface so
  the outcome is defensible either way.

Branch resolution order: α (structural) → β (resistant $0 headroom → earn/no-earn) →
(only if β earns AND is run AND is real-but-ambiguous) exposed control → γ throughout.

---

## § 2 — Hosted controller-native mechanism slate (LOCKED before results)

New module: `coordpy/controller_native_code_mechanism_v1.py` (explicit-import only).
It composes the underused arsenal onto the official ICPC code path. Three candidates:

* **C1 — role-specialized planner / controller.**
  Shared prefix + per-role blocks via `hosted_cache_aware_planner_v12`
  (`plan_per_role_ten_layer_rotated`); roles = {solver, debugger, verifier, patcher}.
  An explicit controller chooses per round among {DRAFT, PATCH, ABSTAIN} from the
  verifier's PUBLIC-sample verdict. Control flow = a role roster + planner, NOT a linear
  reflexion chain.

* **C2 — router-selected multi-candidate controller.**
  Multiple candidate drafts; each scored by a hidden-test-BLIND scorer (public-sample
  pass count → self-consistency cluster → logprob/length proxy). The router
  (`hosted_logprob_router_v12` abstain floor + `hosted_router_controller_v12` decision)
  selects which candidate(s) receive the (expensive) secret-grader budget and commits
  the top one. Selection plane = a router, NOT first-pass-among-K.

* **C3 — tool-substrate audited repair loop.**
  Every grader/executor call is a first-class **audited** `ToolCallSchemaV1` /
  `ToolResultSchemaV1` event in a `ToolAuditChainV1` (`tool_call_substrate_v1`). Retry
  policy is **controller-driven** off the typed executor failure digest
  (`executor_grounded_patcher_v1.parse_failure_digest_v1`): route **PATCH** (minimal,
  conditioned on the latest candidate only) vs **REPLAN** (fresh draft; on parse/import
  error or a repeated-identical digest) vs **ABSTAIN** (low budget + repeated-identical
  digest). Tool-use plane = audited substrate; retry policy = digest-routed; NOT prose
  reflexion text chaining.

**Genuinely-controller-native requirement (all candidates).** A candidate counts as a
real new mechanism only if it exhibits ≥ 2 of: (i) non-linear action routing over
{DRAFT, PATCH, REPLAN, ABSTAIN} (reflexion uses only DRAFT); (ii) a first-class audited
tool plane (reflexion has none); (iii) a failure-digest-conditioned retry policy
(reflexion's next attempt is unconditional). A candidate that collapses to a linear
DRAFT-only chain is **fake-different == reflexion** and is killed (§ 4).

**Never reads the hidden test.** No candidate's model-facing prompt may contain any
secret `.in`/`.ans` byte. The secret grader (`grade_on_secret_v1`) is the FINAL scorer of
the controller's committed choice only; all in-loop feedback is PUBLIC (sample-case
results + judge verdict bit + executor stderr tail), exactly as W120 reflexion B received.
This is the property that makes the mechanism hosted-translatable (text-level, no
hidden-state dependence).

---

## § 3 — Resistant field + replay corpus (LOCKED)

* **Default main field = the W120 resistant ICPC battlefield.** Do NOT invent a new
  family. Deterministic core 30-slice, CID prefix `01bf9ef869a56e20` (asserted). Packages
  = official ICPC org surfaces (RMRC 2024-25 / 2025-26, ECNA NA-East 2024-25 / 2025-26),
  every problem dated strictly after Maverick's KNOWN Aug-2024 cutoff (2024-08-31).
* **Replay corpus = the already-paid real Maverick generations** on disk:
  `results/w120/icpc_pilot/…/icpc_reflexion_calls.jsonl` (330 records = 30 problems × 11
  calls = [A0(T=0) | A1×5(T=0.7) | B×5(T=0.7)]; deterministic record→problem mapping
  `i → i//11`). Re-grading the recovered `response_text` on the official secret + sample
  cases is local subprocess execution = **$0 NIM**.
* **W120 resistant baseline (prior result, cited).** A0 20.00% (6/30), A1 23.33% (7/30),
  B 23.33% (7/30), **B−A1 +0.00pp**; MLB-1 83.33% PASS, MLB-2 8.00% FAIL; verdict FAIL
  `BOUNDED_CEILING_HOLDS_ON_RESISTANT_ICPC`. Pool-union(A0,A1,B) on secret = 8/30 (prior).

---

## § 4 — Structural "fake-different vs real" test (Lane α, LOCKED)

For each candidate, emit a decision-trace structural fingerprint and compare to reflexion B:

* `n_distinct_action_types` (B = 1: DRAFT only).
* `has_audited_tool_plane` (B = false).
* `retry_is_digest_conditioned` (B = false).
* `control_flow_is_linear_chain` (B = true).

A candidate is **REAL (controller-native)** iff it satisfies the ≥2-of-3 requirement in
§ 2 AND `control_flow_is_linear_chain == false`. Otherwise **FAKE_DIFFERENT** → killed,
recorded, not advanced to Lane β. **Lead selection:** among REAL candidates, the lead is
the one that (a) maximises distinct routing events on the resistant replay structure and
(b) most directly maps to a hosted text-API translation (no hidden-state dependence).
Ties broken toward C3 (the audited-tool-plane + digest-router superset of the W124 M6
contract and the W111 M3 patcher lineage). Killed candidates are reported with the reason.

---

## § 5 — NIM-free contract checks (Lane α gate; ALL must PASS)

1. **Tool-call audit chain re-hashes.** The controller's `ToolAuditChainV1.merkle_root()`
   recomputes byte-identically from the persisted (call_cid, result_cid) steps; idempotent
   re-commit of an identical call is refused (`already_committed`); a tampered result byte
   flips the recomputed root. PASS iff root reproduces and tamper is detected.
2. **Official grader-call capture + never-reads-secret.** Every secret-grader invocation is
   a captured audited tool event; a static+dynamic check confirms NO secret `.in`/`.ans`
   byte ever enters a model-facing prompt. PASS iff capture is complete AND the
   secret-leak check is clean.
3. **Role-plan / routing determinism.** Same (problem, candidate pool, digest sequence) →
   identical plan CID and identical routing-decision-sequence CID across two independent
   runs. PASS iff CIDs match.
4. **Same-budget accounting.** The controller's model-generation budget `n_model_calls`
   is exactly the A1/B budget K (= 5) per problem (no extra generations); secret-grader
   budget is accounted explicitly and is the same final-scoring discipline as A1/B
   (grade the committed answer; in-loop grading is PUBLIC-sample only). PASS iff
   `n_model_calls == K` and the budget ledger balances.

Contract verdict = AND of the four. A FAIL here blocks Lane β entirely (no earn).

---

## § 6 — Resistant headroom probe + pilot earn rule (Lane β, LOCKED)

The $0 replay grades each of the 11 per-problem generations on secret + public samples
(official grader) and computes, on the W120 resistant 30-slice:

* `a1_pass5` (== W120 A1 per-problem set), `pool_union_secret` (A0∪A1∪B on secret).
* **`blind_selection_headroom`** = # problems where `a1_pass5 == FAIL` but a hidden-test-
  BLIND selection policy (max public-sample-pass-count → self-consistency majority →
  logprob/length proxy) commits a secret-PASSING pool candidate. (The ONLY headroom a
  pool-replay can show; an oracle upper bound is `pool_union_secret − a1_pass5`.)
* **`reflexion_divergence`** = # A1-fail problems whose B chain shows a stuck signature
  (≥2 repeated candidate-code SHAs OR ≥2 identical failure digests across attempts) where
  C3's digest-router would deterministically choose REPLAN/ABSTAIN instead of another
  near-identical patch.

**Pilot earn gate (E1 ∧ E2 ∧ E3 — all required):**

* **E1** — all four § 5 contract checks PASS.
* **E2** — the lead candidate is REAL (controller-native), not FAKE_DIFFERENT (§ 4).
* **E3 (broad resistant verdict-changing power)** — BOTH:
  * **E3a** `blind_selection_headroom ≥ 2` distinct problems (≥ +6.67pp on the existing
    corpus, strictly past the ±3.34pp null band; ≥2 ⇒ not rescue-concentrated), AND
  * **E3b** `reflexion_divergence ≥ 3` distinct problems (a concrete, enumerated reason a
    controller's NEW trajectory would diverge from the stuck reflexion chain).

If E1∧E2∧E3 hold → **a fresh resistant Maverick controller pilot is EARNED** (§ 7).
If E3 fails (headroom null / thin / rescue-concentrated) → **NOT earned → $0 NIM, no
fresh pilot.** The resistant field is generation-capped for $0 re-routing; ship the
contract-verified mechanism and register `W125-L-RESISTANT-GENERATION-CAP`. A close edge
is NOT sufficient. Reflexion's measured +0.00 on this field is the standing prior; a fresh
pilot must be precursor-earned, not hope-funded.

---

## § 7 — Same-budget accounting + fresh-pilot spend rule (LOCKED)

* **Same budget** (verbatim W120/W121 discipline): same model
  (`meta/llama-4-maverick-17b-128e-instruct`) × same official ICPC package family × same
  official secret grader × same evaluator (verbatim W108 `_mlb_rates` +
  `_evaluate_phase2_gates`) × same K=5 budget on A1 and the controller arm (byte-exact,
  no early-stop credit, no selective retries). The controller arm `B_ctrl` consumes
  exactly K=5 model generations per problem and commits ONE final answer; `final_passed`
  ⟺ that committed answer passes EVERY secret case. A1 = first-pass-among-K (pass@K).
* **Gates** (verbatim): MLB-1 invocation floor 0.33; MLB-2 rescue floor 0.33; the 9
  Phase-2 composite gates; retirement margin **+5.00pp**; null band **±3.34pp** at n=30.
* **Fresh-pilot spend is EARNED iff** § 6 E1∧E2∧E3 hold. Sequence then = canary (harness
  validation only) → cheapest honest resistant pilot = 1 seed × 30 × K=5 = **330 NIM
  calls** → same-budget A1 comparator on the SAME resistant field.
* **No fresh-pilot spend** if E3 fails. No new n=30 seed-chasing by default. No stronger-
  model spend unless § 9 opens. No 405B. No reopening MBPP+ V2 / frozen cross-modal /
  closed Llama-3.1 rescue / APPS main-lane NIM. No dirty exposed benchmark sold as a win.
  A close local/hosted edge is NOT sufficient to justify spend.

---

## § 8 — Exposed-control earn / no-earn rule (LOCKED)

* If the resistant pilot is **NOT earned** (E3 fail) → **do NOT** buy an exposed control.
* If the resistant pilot **FAILs cleanly** when run → stop; do NOT auto-buy exposed.
* If the resistant pilot **PASSes strongly / shows a real same-budget gain** → THEN earn
  the matched exposed ICPC control (W121 family) on the SAME controller stack.
* If the resistant result is **close/ambiguous** → buy exposed ONLY if it has real
  interpretation-changing power (resolves mechanism-vs-exposure); do NOT default to two
  pilots just because both fields exist.
Resistant-first is the frontier move; exposed is worth buying only when the new mechanism
gives something real to interpret.

---

## § 9 — Per-model disclosure status + certification rule (Lane γ, LOCKED)

Reuse `coordpy.stronger_model_cutoff_certification_v1` (C1∧C2∧C3∧C4; decision CID
`258b6ed7`, invariant W114→W124). Re-check PRIMARY sources for: Maverick,
Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B-2603, GLM-5, and any newly
reachable same-budget-comparable model. A model SUPERSEDES Maverick as the hosted target
ONLY if it becomes primary-KNOWN (disclosed cutoff) AND certifiable on the matched ICPC
family (resistant side needs a KNOWN cutoff ≤ ~Aug-2024). Standing prior:
**{KNOWN:1 (Maverick, Aug-2024), UNKNOWN:4}** ⇒ Maverick is the only certifiable hosted
target. The local transformer-native line stays CLOSED unless new local code-model supply
appears (do NOT reopen W124 by default).

---

## § 10 — Carry-forward registration (LOCKED shape; filled from JSON)

* W89 (+5.56) + W105 (+7.00) remain the only two retirements unless a fresh resistant
  pilot is earned, run, and clears the +5.00pp clean-superiority bar.
* If E3 fails: register **`W125-L-RESISTANT-GENERATION-CAP`** — the resistant ICPC field
  is generation-capped for $0 controller re-routing (pool-union reaches 8/30, A1 captures
  7, residual ≤ +1 within the null band); the controller-native arsenal is real and
  contract-verified but its potential value lives only in NEW trajectories on the
  uniformly-unsolved problems, unfundable without a precursor signal. This is the
  MECHANISM-LEVER sibling of W123's battlefield-supply cap and W124's local-encoder cap.

---

## § 11 — graphify deliverables (LOCKED)

* graphify refreshed at START (`graphify update .`; built from HEAD `92ccccfc`) and END
  (after material code/doc changes).
* Required commands run + reported: `graphify explain` on the six arsenal nodes +
  `coordpy_icpc_public_functional_v1`; `graphify path hosted_router_controller_v12
  tool_call_substrate_v1`; `graphify affected hosted_router_controller_v12.py`;
  `graphify query` as a secondary claim-surface finder.
* The new module must create the first semantic bridge between the hosted-controller stack
  and the tool-call substrate (currently 6 hops apart with no direct edge) — the END graph
  must show the new edges.

---

## § 12 — W126 branch logic (pre-committed)

* If W125 β is **NOT earned** (expected if E3 null): W126 = either (a) accept the standing
  bounded HumanEval-family ceiling (W89+W105) + the registered generation-cap, OR (b) an
  **operator-greenlit** fresh hosted controller pilot funding the NEW-trajectory hope on
  the 22 uniformly-unsolved resistant problems (explicitly flagged NOT precursor-earned),
  OR (c) a primary-KNOWN reachable stronger-than-Maverick model opens § 9, OR (d) a
  code-competent LOCAL model becomes loadable (reopens W124 M4 + lets the controller's
  gain be shown locally at $0).
* If W125 β **earns and the fresh pilot is run**: W126 = carry the verdict — retire iff
  `PASS_MECHANISM_DRIVEN` ≥ +5.00pp clean (9/9 gates + MLB-1 + MLB-2); else register the
  controller-pilot cap and, if real-but-ambiguous, earn the exposed control.
* `COO-9` stays the lead path unless the evidence forces a different code-line move.
