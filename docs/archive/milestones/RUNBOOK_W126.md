# RUNBOOK W126 — Family-adapted repair synthesis on resistant ICPC + fresh hosted pilot only if the new-trajectory line earns it

**Status:** LOCKED pre-registration (written BEFORE any NIM call and BEFORE interpreting
the precursor verdict). Decision authority = the pre-committed code rules in
`coordpy/family_adapted_repair_synthesis_v1.py`; this document is their human surface.
`ultracode` OFF. Stable boundary: `coordpy.__version__ == "0.5.20"`,
`SDK_VERSION == coordpy.sdk.v3.43`, no PyPI, `coordpy/__init__.py` untouched, advanced
work explicit-import-only.

## § 0 — Why W126 is NOT another lap

W120–W125 closed three cheap levers and one mechanism question:

- **battlefield** (W120 resistant +0.00 FAIL / W121 exposed +3.33 FAIL / W122 n=30
  unresolvable / W123 ≥100-per-field supply-UNREACHABLE),
- **local encoder** (W124 distilgpt2 hidden state adds nothing; no code-competent local
  model),
- **$0 re-routing** (W125: the hosted controller stack is REAL + contract-clean, but the
  resistant field is GENERATION-CAPPED — `blind_selection_headroom = 0`, pool-union 8/30,
  the 22 unsolved problems uniformly unsolved across all 11 generations).

W125 answered "can the controller CHOOSE better among the same 11?" — no. W126 asks the
unanswered question: **can CoordPy SYNTHESIZE genuinely NEW resistant-code candidates from
the already-paid pool plus an official-family teacher corpus, without leaking target
answers?** This is new-trajectory creation, not reranking. It is the only honest lever left
that W125 explicitly did not pull. W126 is NOT: more battlefield, more local-encoder
probing, more reranking, bounded-context / compaction / summarization tricks (those remain
anti-patterns).

## § 1 — α / β / γ branch logic (pre-committed)

- **Lane α (MAIN, NIM-free):** build the family-adapted synthesis slate (S1/S2/S3/S-CONS,
  optional S4) on the W120 resistant 30-slice; measure the ORACLE ceiling (does any new
  trajectory pass secret) and the BLIND headroom (does a hidden-test-blind committed
  trajectory pass secret) on the 22 uniformly-unsolved problems. Lane α succeeds only if it
  either **creates a precursor signal** or **proves exactly why synthesis is dead** — "no
  signal" alone is not a complete lane.
- **Lane β (resistant-first, conditional spend):** apply the precursor earn gate (§ 5). A
  fresh hosted Maverick resistant pilot is run **iff** P1 ∧ P2 (§ 5–6). Else **$0 NIM**,
  register the synthesis cap. Exposed control is downstream of an informative resistant
  result only (§ 7).
- **Lane γ (truth / gate / graphify):** re-check primary-source cutoffs (§ 8); keep the
  W123 battlefield-supply cap and W124 local-encoder cap closed unless new evidence changes
  them; refresh graphify START + END (§ 9); land executable code (not docs only).

## § 2 — No-leakage rule (LOCKED, enforced by `SynthesisLeakageGuardV1`)

1. NEVER expose a target problem's accepted solution, secret input, secret answer, or
   validator internals to any model-facing prompt or synthesis logic.
2. The synthesiser NEVER opens a resistant target's `submissions/` or `data/secret/`.
   Resistant target packages are used ONLY for grading, package completeness, and static
   metadata (statement + public `data/sample/`).
3. Every synthesized candidate passes `SynthesisLeakageGuardV1.check`: it must carry no
   secret `.in`/`.ans` byte-run (≥3 chars, secret-only) and no full secret-case
   concatenation and no line of the target's own accepted solution — UNLESS that run
   already appears in the PROVENANCE (the source generations + public samples that
   synthesis recombines). A secret run present in a candidate but ABSENT from provenance is
   the real injection signature (an accidental secret-file read); a run already in
   provenance is a base-model coincidence (the model wrote it WITHOUT secret access — e.g.
   an emoticon literal that coincides with a short secret answer) and is NOT a leak. A
   planted secret/answer absent from provenance is a verified positive control (the guard
   must bite).
4. If ANY synthesis input fails the guard ⇒ verdict `SYNTHESIS_INVALID_LEAKAGE`, the lane is
   killed, **$0 NIM**.

## § 3 — Family-level teacher-corpus rule (LOCKED)

- The teacher corpus is the EXPOSED-side (pre-cutoff, W121 family) accepted Python
  solutions only, loaded by `load_exposed_teacher_corpus_v1` from `/tmp/w121_icpc/**/
  submissions/accepted/*.py`.
- It is used ONLY as a FAMILY-LEVEL motif/idiom prior (derived by `derive_family_motifs_v1`),
  never as same-problem answer material.
- Disjointness is asserted: any teacher solution whose problem short-name equals a resistant
  target short-name is dropped. Corpus identity is pinned by `corpus_cid`.

## § 4 — New-trajectory synthesis slate (LOCKED before results)

All operators are deterministic text/AST transforms; they never execute code outside the
audited official grader. Merely reranking / relabeling reflexion / majority-voting pass-fail
is INSUFFICIENT (§ 0) — each operator must produce a NEW program or a NEW per-case decision.

- **S1 cross-candidate splice** (`synth_splice_v1`): AST recombination of the 11 generations
  — swap a same-named function body from a donor generation into a host generation ⇒ new
  programs that fuse one generation's structure with another's sub-routine.
- **S2 digest-grounded repair** (`synth_digest_repair_v1` + `parse_failure_digest_v1`):
  the typed PUBLIC executor digest routes principled micro-repairs (robust stdin tokenizer,
  `setrecursionlimit`, entrypoint-call insertion, yes/no token-casing) — NO answer-fudging
  (no numeric output tweaking).
- **S3 family-motif harden** (`synth_motif_harden_v1`): family idioms (fast buffered I/O,
  recursionlimit) derived from the exposed teacher corpus, applied as hardening transforms.
- **S-CONS output-consensus dispatcher** (`eval_output_consensus_v1`): a NEW program that,
  per input, runs the trusted generations and emits the trust-weighted plurality output;
  trust weights come from `trust_weighted_consensus_v1` (the adversarial-consensus arsenal)
  on the blind generation scores. Three BLIND variants (majority / trust-weighted /
  sample-passers). This is the lever W125 could not express: it can pass a problem whose
  generations are each correct on a DIFFERENT subset of cases.
- **S4 learned repair-action policy** (conditional): only if the labelled corpus is large
  enough; else NOT_WARRANTED (W124 precedent: chance on n≈14 events). Uses
  `constrained_policy_optimisation_v1` / `learned_economics_controller_v1`.

The sharp precursor slice = the **22 uniformly-unsolved** resistant problems (W125;
re-derived from the W126 grade cache). Selection is hidden-test-BLIND; the official secret
grader scores only the committed answer; the synthesis trace CID is deterministic.

## § 5 — Precursor earn / no-earn rule (LOCKED; `apply_synthesis_earn_gate_v1`)

Definitions (on the 22 uniformly-unsolved problems):

- `oracle_new_solved` = # problems where ANY synthesized trajectory (S1/S2/S3 program or any
  blind S-CONS variant) passes ALL official secret cases.
- `blind_new_solved` = # problems where a HIDDEN-TEST-BLIND committed trajectory passes ALL
  official secret cases: a top-`B_syn` (=5) blind-ranked S1/S2/S3 program, or a blind S-CONS
  variant. This is the pilot-earning metric.
- `distinct_families` = the set of winning families (consensus variant / program op) over
  the blind wins.

Gates:

- **P1** = `blind_new_solved ≥ 2` (≥ 2 DISTINCT problems unsolved by the entire old pool,
  now blind-solved by synthesis).
- **P2** = `|distinct_families| ≥ 2` (the wins are not a single narrow trick — ≥ 2 distinct
  failure families / package surfaces / op-or-digest types).
- **Leakage** = every synthesized candidate is guard-clean (§ 2).

Verdict:

- `leakage ∧ P1 ∧ P2` ⇒ `FRESH_RESISTANT_PILOT_EARNED_SYNTHESIS_HEADROOM`.
- `leakage ∧ (oracle_new ∨ blind_new) ∧ ¬(P1∧P2)` ⇒
  `FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_THIN` (real but sub-threshold headroom; $0).
- `leakage ∧ oracle_new = 0` ⇒ `FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_DEAD`
  (deterministic recombination/repair of capability-failed generations cannot manufacture
  correct algorithms at $0; $0).
- `¬leakage` ⇒ `SYNTHESIS_INVALID_LEAKAGE` (lane killed; $0).

A close blip, a same-problem leak, or an oracle-only ceiling does NOT count as a win.

## § 6 — Resistant-first pilot spend rule (LOCKED)

- Fresh hosted spend is earned ONLY by `FRESH_RESISTANT_PILOT_EARNED_SYNTHESIS_HEADROOM`
  (P1 ∧ P2 ∧ leakage-clean).
- If earned: run the cheapest honest fresh hosted resistant pilot on the full W120 resistant
  30-slice, same budget shape as W120/W125 (Maverick, same official grader, same K, same
  evaluation rules), with the controller + synthesis line integrated honestly and the
  synthesis trace CID recorded. Same-budget accounting: model-generation budget bounded;
  exactly the locked grade discipline.
- If NOT earned: **$0 NIM**, no fresh resistant pilot; register the synthesis cap
  carry-forward.
- No new n=30 seed-chasing by default. No stronger-model spend unless § 8 opens.

## § 7 — Exposed-control earn / no-earn rule (LOCKED)

- The matched exposed ICPC control is NOT automatic. Buy it ONLY if the fresh resistant pilot
  is run AND produces real interpretation-changing value.
- If the resistant pilot is not earned (§ 5) ⇒ exposed control NOT earned and NOT bought
  (resistant-first).

## § 8 — Per-model disclosure status + certification rule (Lane γ, LOCKED)

- Re-check primary-source cutoff disclosures for: Maverick, Qwen3-Coder-480B,
  DeepSeek-V4-pro, Mistral-Small-4-119B-2603, GLM-5, and any newly reachable
  same-budget-comparable model, via `stronger_model_cutoff_certification_v1`
  (C1∧C2∧C3∧C4 gate; decision CID `258b6ed7`).
- A stronger-than-Maverick model can supersede Maverick ONLY if it becomes primary-KNOWN and
  certifiable on the ICPC family (cutoff ≤ the resistant instrument frontier). Otherwise
  Maverick remains the hosted target; the gate stays CLOSED (`{KNOWN:1, UNKNOWN:4}`).
- No 405B expensive run unless reachability changes and a pre-committed gate clears.

## § 9 — graphify deliverables (LOCKED)

- Refresh `graphify update .` at START and END; record the END HEAD.
- `graphify explain` on the mined arsenal: `controller_native_code_mechanism_v1`,
  `executor_grounded_patcher_v1`, `adversarial_consensus_repair_v1`,
  `compose_repair_integrity_pipeline_v1`, `constrained_policy_optimisation_v1`,
  `learned_economics_controller_v1`, `run_icpc_public_construction_v1`.
- `graphify path controller_native_code_mechanism_v1 compose_repair_integrity_pipeline_v1`
  + `graphify affected` on the new module — record the new bridge edge the W126 module
  creates (the synthesis module imports the repair/consensus arsenal onto the ICPC path,
  which graphify showed were otherwise only trivially connected).

## § 10 — Carry-forward registration (LOCKED shape; filled from JSON)

- Retirements UNCHANGED unless a pilot earns and clears the retirement bar: **W89 (+5.56) +
  W105 (+7.00)** remain the only two confirmed retirements until proven otherwise.
- On a NOT_EARNED verdict, register `W126-L-RESISTANT-SYNTHESIS-CAP` (the SYNTHESIS-lever
  sibling of W123 battlefield-supply / W124 local-encoder / W125 re-routing caps): the
  resistant field's 22 uniformly-unsolved problems are capability failures — deterministic
  recombination/repair/consensus over the already-paid generations cannot manufacture
  correct algorithms at $0 without new model capability.
- Named claims filled ONLY from the emitted `synthesis_precursor_verdict.json`.

## § 11 — W127 branch logic (pre-committed)

- If W126 EARNED + the pilot cleared the retirement bar on resistant ICPC ⇒ W127 =
  confirm (multi-seed / cross-scale) the third retirement.
- If W126 EARNED + pilot FAILED ⇒ W127 = the same hardened-ceiling conclusion as a
  null resistant pilot (the synthesis raised headroom but Maverick still could not realize
  it online) — accept the bounded ceiling or pursue a genuinely different axis.
- If W126 NOT_EARNED (synthesis thin or dead) ⇒ W127 = accept the bounded HumanEval-family
  ceiling + the registered synthesis cap; fire only on (a) a code-COMPETENT local model
  that lets new trajectories be generated at $0, (b) an OPERATOR-GREENLIT fresh hosted
  pilot explicitly funding the new-trajectory hope (NOT precursor-earned), or (c) a
  primary-KNOWN reachable stronger-than-Maverick model certifiable on the ICPC family.
- `COO-9` stays the lead unless the evidence genuinely forces a different code-line move.
