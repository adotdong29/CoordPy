# RESULTS — W126: family-adapted repair synthesis on resistant ICPC (3 lanes)

**Date:** 2026-06-01 · executes the pre-committed `docs/RUNBOOK_W126.md` α/β/γ branch
logic, locked BEFORE the precursor was interpreted. `ultracode` OFF. Decision CID
`258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched.

> **Verdict (Lane α/β):** `FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_DEAD` — oracle
> ceiling **0/22**, blind **0/22**, leakage-clean across all 22; **$0 NIM**, no fresh
> pilot, no exposed control. **W89 (+5.56) + W105 (+7.00) STAND as the only two
> retirements.** New carry-forward `W126-L-RESISTANT-SYNTHESIS-CAP`.

## Why W126 mined synthesis, not re-routing

W120–W125 closed three cheap levers + answered one mechanism question:

| milestone | lever | result |
|---|---|---|
| W120/W121/W122/W123 | battlefield | resistant +0.00 / exposed +3.33 FAIL; n=30 unresolvable; ≥100-per-field supply-UNREACHABLE |
| W124 | local encoder | distilgpt2 hidden state adds nothing; no code-competent local model |
| W125 | $0 controller **re-routing** | controller stack REAL + contract-clean, but `blind_selection_headroom = 0`, pool-union 8/30 → GENERATION-CAPPED |

W125 answered "can the controller **choose** better among the same 11 generations?" — no.
W126 asks the unanswered question: **can CoordPy *synthesize* genuinely NEW resistant-code
candidates from the already-paid pool + an official-family teacher corpus, without leaking
target answers?** This is new-trajectory creation. It is the only honest lever W125 did not
pull.

## The sharp precursor slice (NIM-free grade-cache recon)

`scripts/run_w126_grade_cache_recon_v1.py` re-graded all 330 already-paid Maverick
generations (11 per problem) on the official secret + sample cases (`$0 NIM`):

- **8 solved / 22 unsolved** (reproduces W125 pool-union 8/30 exactly).
- Of the 22 unsolved: **19 VISIBLE** (no generation passes all public samples ⇒ a
  wrong-algorithm failure with a visible signal) + **3 HIDDEN** (`enchantedmaze`, `genies`,
  `spiesvsspies` — ≥1 generation passes all public samples but fails secret ⇒ a subtle
  edge-case bug; the only place a blind signal is absent).
- Generation digest distribution across the 330: **80 ok / 241 wrong / 9 Timeout** — only
  9 TLEs ⇒ the family-motif fast-I/O / recursionlimit hardening (S3) addresses very few;
  the 22 unsolved problems are dominated by visible wrong-algorithm failures with **10–11
  DISTINCT generations per problem** (high diversity, uniformly wrong).

This already frames the honest expectation: the 22 are **capability failures**, not surface
bugs — the question is whether deterministic recombination/repair/consensus can manufacture
a correct trajectory from wrong ones at $0.

## Lane α — the new-trajectory synthesis slate (NIM-free)

`coordpy/family_adapted_repair_synthesis_v1.py` (explicit-import-only) wires the unused
repair/consensus arsenal onto the official-ICPC path (graphify: the controller path and
`compose_repair_integrity_pipeline_v1` were 4 hops apart through a trivial `str` node — NO
real semantic edge; this module is the bridge):

- **S1 cross-candidate splice** — AST function-body recombination across the 11 generations.
- **S2 digest-grounded repair** — `parse_failure_digest_v1` typed digest routes principled
  micro-repairs (robust stdin tokenizer / `setrecursionlimit` / entrypoint-call / yes-no
  casing); NO answer-fudging.
- **S3 family-motif harden** — fast-I/O + recursionlimit idioms mined from the EXPOSED-side
  accepted solutions of OTHER problems (108 accepted `.py` / 38 problems, problem-disjoint;
  `corpus_cid 42d108d4…`).
- **S-CONS output-consensus dispatcher** — a NEW program that, per input, runs the trusted
  generations and emits the **trust-weighted plurality** output (trust from
  `adversarial_consensus_repair_v1.trust_weighted_consensus_v1` on the blind generation
  scores); three BLIND variants (majority / trust-weighted / sample-passers). This is the
  lever W125 could not express: it can pass a problem whose generations are each correct on
  a DIFFERENT subset of cases. (Tested: it recovers a minority-correct generation under
  trust-weighting where naive majority FAILS — the arsenal is load-bearing.)

All execution is via a hardened capped runner (bounded stdout + process-group kill ⇒ a
pathological synthesized candidate cannot blow up or orphan). A safe capped screen is
CONFIRMED by the official `grade_on_secret_v1`. Selection is hidden-test-BLIND; the secret
grader scores only the committed answer; the synthesis-trace CID is deterministic.

## No-leakage rule (enforced)

`SynthesisLeakageGuardV1`: every synthesized candidate is checked against the target's
secret `.in`/`.ans` byte-runs + full secret-case concatenations + the target's own
accepted-solution lines; the synthesiser never opens a resistant target's `submissions/` or
`data/secret/`. Positive control verified (a planted secret/answer is caught). Teacher
corpus is EXPOSED-side, problem-disjoint (asserted). `leakage_all_clean` is a gate input.

## Verdict — Lane α/β (`synthesis_precursor_verdict.json`)

On the 22 uniformly-unsolved resistant problems, the slate synthesized **894 new candidate
programs** (avg 40.6/problem — S1 splice + S2 digest-repair + S3 motif-harden) and ran the
S-CONS dispatcher in 3 blind variants (66 evaluations), graded on the official secret +
sample cases ($0 NIM, ~19 min):

| quantity | value | reading |
|---|---|---|
| synthesized candidates | **894** | a serious new-trajectory attempt, not a token gesture |
| **oracle ceiling** (any synth program OR any consensus variant passes ALL secret) | **0 / 22** | even WITH the oracle picking the best, NO new trajectory is correct |
| **blind P1** (top-`B_syn`=5 committed program OR blind consensus passes secret) | **0 / 22** | the pilot-earning metric — zero |
| distinct winning families | **[]** | P2 vacuous |
| leakage-clean | **22 / 22** | every candidate provenance-clean (incl. the emoticons coincidence — see below) |
| trace CIDs | **22 distinct, deterministic** | synthesis is reproducible |

**Earn gate (RUNBOOK § 5):** leakage-clean ✓ ∧ P1 (`blind_new ≥ 2`) ✗ (= 0) ∧ P2 ✗ ⇒
**`FRESH_RESISTANT_PILOT_NOT_EARNED_SYNTHESIS_DEAD`**. The oracle ceiling being **0** is the
sharp finding: this is not "blind selection missed latent headroom" (W125's shape) — there
is **no headroom to miss**. Deterministic recombination (S1), digest-grounded repair (S2),
family-motif hardening (S3), and even trust-weighted per-case output consensus (S-CONS — the
lever W125 structurally could not express, which can pass a problem whose generations are
each correct on a different subset of cases) ALL produce zero secret-passing trajectories.
The 22 problems are **capability failures** (Maverick-17B never produced the right algorithm
in any of its 11 attempts, and the 10–11 distinct-but-uniformly-wrong generations share no
complementary-correct fragments to fuse); $0 deterministic synthesis over wrong algorithms
cannot manufacture a correct one.

**The emoticons leakage check (a verified false-positive correction).** On the first run the
guard flagged `emoticons` because three of its 3-char secret answers (`^_^`, `^o^`, `^^;`)
appear as string literals in the model's OWN generations — provenance-clean coincidences
(the base model wrote them without secret access). The guard was corrected to flag a secret
run only when it is ABSENT from the provenance (source generations + samples) — i.e. a real
injection signature — which preserved the tripwire (a planted secret absent from provenance
still bites; tested) while eliminating the coincidence. Final run: 22/22 leakage-clean.

**Lane β (resistant-first spend):** NOT earned ⇒ **$0 NIM**, no fresh resistant pilot. The
new-trajectory hope on the 22 unsolved problems requires NEW model generation, which the $0
precursor is structurally unable to supply a signal for — exactly because the failures are
capability failures with no hidden-test-blind discriminator. **Exposed control NOT earned
and NOT bought** (resistant-first; RUNBOOK § 7).

## Lane γ — stronger-model gate / truth

`stronger_model_cutoff_certification_v1` re-affirmed **`NO_CERTIFIABLE_STRONGER_MODEL`**,
decision CID **`258b6ed7…` invariant**, registry `{KNOWN:1, UNKNOWN:4}` (Maverick KNOWN
Aug-2024 certifiable-but-settled; Qwen3-Coder-480B / DeepSeek-V4-pro /
Mistral-Small-4-119B-2603 / GLM-5 UNKNOWN-from-primary). No new primary-source disclosure
since W125 (same day). The W123 battlefield-supply cap and W124 local-encoder cap stay
closed. The W126 spend gate is Lane β (synthesis headroom), not Lane γ.
`results/w126/stronger_model_gate/gate_recheck_v1.json`.

## Carry-forward

Exactly **TWO** confirmed retirements stand — **W89** (base HumanEval × llama-3.3-70b,
+5.56 pp) and **W105** (HumanEval+ × llama-3.3-70b, +7.00 pp), both contamination-EXPOSED
HumanEval-family at 70B. W126 **retires none and adds none**.

New limitation `W126-L-RESISTANT-SYNTHESIS-CAP` — the SYNTHESIS-lever sibling of the cap
taxonomy: W123 **battlefield-supply** cap (no ≥100-per-field matched battlefield) → W124
**local-encoder-supply** cap (no code-competent local model) → W125 **re-routing** cap
(`blind_selection_headroom = 0` over the 11 generations) → **W126 synthesis cap** (oracle
ceiling 0: deterministic recombination/repair/consensus over the already-paid generations
cannot create a single new secret-passing trajectory on the 22 uniformly-unsolved resistant
problems). The honest sharp negative on the spend question (no precursor headroom) alongside
a genuinely-mined, contract-clean, leakage-clean, arsenal-native slate (the repair/consensus
arsenal was bridged to the ICPC path for the first time).

## Named claims (THEOREM_REGISTRY)

- `W126-L-RESISTANT-SYNTHESIS-CAP` (empirical): on the W120 resistant 30-slice's 22
  uniformly-unsolved problems, a family-adapted synthesis slate (S1 splice / S2
  digest-repair / S3 exposed-motif harden / S-CONS trust-weighted output consensus)
  synthesizing 894 new leakage-clean candidate programs reaches an **oracle ceiling of 0**
  (no synthesized trajectory passes the official secret cases) ⇒ a fresh hosted pilot is not
  precursor-earned; the resistant field is synthesis-capped for $0 new-trajectory creation.
- `W126-T-SYNTHESIS-OF-CAPABILITY-FAILURES-IS-DEAD-NOT-BLIND-CAPPED` (empirical): the
  ceiling is 0 under ORACLE selection, not merely under blind selection ⇒ the cap is an
  absence of headroom (capability failure), distinct from W125's blind-selection cap over
  existing-but-correct generations.

## Artifacts

- `coordpy/family_adapted_repair_synthesis_v1.py` (explicit-import-only; leakage guard +
  teacher corpus/motifs + S1/S2/S3/S-CONS + capped runner + earn gate).
- `scripts/run_w126_grade_cache_recon_v1.py`, `scripts/run_w126_synthesis_precursor_v1.py`,
  `scripts/run_w126_stronger_model_gate_recheck_v1.py`.
- `tests/test_w126_family_adapted_repair_synthesis_v1.py` (17 tests; falsifiability-first;
  validated by direct execution — local pytest/attrs env broken).
- `results/w126/cache/grade_cache_v1.json`,
  `results/w126/lane_alpha_beta/synthesis_precursor_verdict.json`,
  `results/w126/stronger_model_gate/gate_recheck_v1.json`.
- `docs/RUNBOOK_W126.md` (pre-registration, locked before the precursor).
