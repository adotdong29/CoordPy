# RESULTS W132 — resistant-by-construction hard-family battlefield + exact-oracle gates + Maverick pilot (NIM-infra-blocked) + DEV_ONLY local characterization

**One line.** W123 bounded the OFFICIAL resistant-package supply and W131 found the model
axis blocked on cutoff DISCLOSURE; both share one upstream dependency — a contamination-
RESISTANT battlefield we either inherit or certify against a disclosed cutoff. **W132
removes that dependency by MINTING the battlefield ourselves**: a CoordPy-owned, exact-
oracle, novelty-guarded, deterministic battlefield of **33 admitted** resistant-by-
construction algorithmic problems, targeted at the failure families that actually beat the
mechanism stack. The battlefield **earned the Maverick pilot cleanly** (≥30, all gates,
resistance-certified). The Maverick frontier pilot is **NIM-infra-blocked this session**
(the NVIDIA endpoint was unreachable — even a 16-token Maverick call timed out at 90 s),
so it is registered as a **push-button** re-run, NOT a science result. A clearly-labelled
**DEV_ONLY** local-Ollama characterization validated the β pipeline end-to-end on the
minted field. **W89 (+5.56) + W105 (+7.00) remain the only two retirements; W132 retires
none.**

All numbers below are filled ONLY from emitted verdict JSON
(`results/w132/**`).

---

## Lane α — resistant-by-construction battlefield (MAIN empirical, $0 NIM) — **SUCCESS**

New modules (explicit-import only):
`coordpy.resistant_by_construction_battlefield_v1` (framework: minted-problem record,
the three-program oracle harness, the six per-problem quality gates, the novelty guard,
the manifest/CID, Maverick resistance certification) +
`coordpy.resistant_by_construction_slate_v1` (the 33-template slate).

Each minted problem ships THREE independent complete stdin/stdout programs so correctness
is machine-checked, never asserted: `ref_source` (the scalable correct program whose
stdout IS the answer key), `brute_source` (an INDEPENDENT obviously-correct exhaustive
oracle, run on the small cases to cross-check `ref_source`), and `naive_source` (the
admissible-wrong trap). The model under test sees ONLY the statement + the PUBLIC samples
and is graded by the audited `grade_icpc_candidate_case_v1` token-diff / float oracle
(exit-0-iff-every-hidden-case-passes, NO LLM judge). Minted problems are emitted as
`IcpcPilotProblemV1` so the *already-validated* W120 reflexion bench consumes them verbatim.

**Build verdict (`results/w132/battlefield/battlefield_verdict_v1.json`; re-derivable
byte-identically):**

| field | value |
|---|---|
| templates minted | 33 |
| gate-passing | 33 |
| **admitted (novelty-clean)** | **33 ≥ 30** ✓ |
| mode histogram | COMPLEXITY_BLIND 9 / HIDDEN_EDGE_STATE_MISS 8 / WRONG_ALGORITHM_ADMISSIBLE 8 / SEARCH_ENUM 8 |
| distinct families | 33 (one per problem) |
| `manifest_cid` | `562aafbd62f550d1…` |
| `raw_cid` | `1e9a2a42f20f05ec…` |
| core 30-slice `core_slice_cid` | `f6a2ebed3da2f13b…` (mode-stratified 8/8/7/7) |
| deterministic regeneration | TRUE (re-mint → byte-identical manifest CID) |
| Maverick resistance | RESISTANT (minted 2026-06-02 > cutoff 2024-08-31 by date AND fresh instances by construction; reused W114 C1..C4 gate corroborates `certifiable`) |
| minted date / seed / exec timeout | 2026-06-02 / 132 / 8.0 s |

**Quality gates (every admitted problem passes ALL):** exact-oracle self-test +
small-vs-large `brute`↔`ref` agreement (non-vacuous); reference solvable in budget;
discriminating-hidden-case (`naive` PASSES every public sample AND FAILS ≥1 hidden case in
its declared mode — TIMEOUT for COMPLEXITY_BLIND, WRONG_ANSWER otherwise); public/hidden
split integrity; deterministic regeneration; pass-fail-only.

**Hard-family target rule (§3 of the runbook), what it selected:** the slate prioritises
the W130/W131 atlas modes — `WRONG_ALGORITHM_ADMISSIBLE` (named-but-wrong greedy where DP
is required: coins / weighted-interval / knapsack / partition / LIS / house-robber /
max-product / LCS), `HIDDEN_EDGE_STATE_MISS` (wrap / overlap / bracket-type / inclusivity
/ tie / sign / sort corners), `COMPLEXITY_BLIND` (O(N²) naive TLEs on the large stress
case while the O(N log N)/O(N) reference finishes), and `SEARCH_ENUM` (small-n exhaustive
oracle exact; naive = ordered-vs-unordered / wrong-recurrence / blocks-ignored miscount).

**Novelty / near-duplicate rule, what it rejected:** `novelty_filter_v1` rejects a problem
iff its statement char-5-gram Jaccard ≥ 0.55 with an already-accepted minted problem, or
its statement embeds an official ICPC identity token (the W120 short-names, paraphrase
guard). On the first build it correctly fired on a near-duplicate (`wa_max_product_subarray`
J=0.745 vs `cb_max_subarray_sum`); the two are genuinely different algorithms (Kadane sum
vs min/max product DP), so the product statement was differentiated and BOTH retained. The
guard is validated by a planted-duplicate positive control + an official-identity control
in the tests.

**No-leakage rule:** `MintedProblemV1.to_pilot_problem` ships ONLY the statement + public
samples + the hidden grader; `ref_source`/`naive_source`/`brute_source` are never in any
model-facing prompt; the bench's reflexion feedback uses ONLY the public samples + judge
verdict bit + executor stderr (verbatim the W120 anti-cheat). A unit test asserts the
model-facing payload contains no solver token tells.

---

## Lane β — generated-family mechanism validation + Maverick pilot

**Earn check (RUNBOOK §8b): the battlefield EARNED the pilot** — ≥30 admitted ✓, all
quality gates pass ✓, Maverick resistance-certified ✓, evaluation rule locked before spend
✓. The arms are the *already-validated* W88/W89 three-arm same-budget mechanism (A0 single-
shot / A1 self-consistency-pass@K / B sequential reflexion) at K=5, 1 seed (132001), graded
by the audited oracle, scored by the verbatim W108 `_mlb_rates` + `_evaluate_phase2_gates`
— the SAME code that scored W89/W105/W120.

**Maverick frontier pilot = NIM-INFRA-BLOCKED this session (operational, NOT a science
result).** The NVIDIA NIM endpoint was unreachable: the background calibration completed
exactly 1 of 66 calls before repeated 240 s read-timeouts, and two isolated probes (a
32-token and a 16-token Maverick chat) both timed out (120 s / 90 s). This is a hardware/
infra outage (the throughput-sibling of W131's CPU-bound local 32B), reported AS such — it
is **not** a `FAIL` and does **not** register `W132-L-RESISTANT-BY-CONSTRUCTION-PILOT-CAP`
(that cap is reserved for an executed `FAIL`). The pilot is fully wired + slice-CID-guarded
+ push-button (`scripts/run_w132_calibration_and_pilot_v1.py --mode pilot`); it re-runs the
instant the endpoint recovers.

**DEV_ONLY local-Ollama characterization (`scripts/run_w132_dev_only_local_characterization_v1.py`;
`results/w132/dev_only_local/`) — STRICTLY NOT a frontier/retirement claim.** Because the
field is resistant *by construction*, an UNKNOWN-cutoff LOCAL model is a legitimate non-
contaminated test (the W132 γ payoff) — but `qwen2.5-coder:7b` is far weaker than the
Maverick target, so this characterizes the INSTRUMENT + the mechanism on a weak local model
and CANNOT retire anything. It confirmed the β pipeline executes end-to-end on the minted
field. **Status: `PARTIAL_THROUGHPUT_LIMITED`** (`results/w132/dev_only_local/.../dev_only_status.json`):
the run completed **8 clean Ollama generation→code-extraction→subprocess-exact-oracle-
grading→A0/A1/B-arm-outcome cycles** on the minted field before it was stopped — concrete
evidence the β pipeline executes end-to-end on the minted corpus. But `qwen2.5-coder:7b`
runs at **~100 s/call** on this host (the throughput-sibling of W131's CPU-bound local 32B),
so a full 6-problem characterization is ~1.8 h and the 30-slice ~9 h — throughput-
impractical. The DEV characterization is therefore bounded; it is NOT a frontier or
retirement result (every field tagged `frontier_claim=false`, `can_retire=false`), and the
Maverick frontier pilot stays the primary (NIM-infra-blocked, push-button).

---

## Lane γ — stronger-model gate / graphify / truth ($0) — gate CLOSED

`scripts/run_w132_stronger_model_gate_recheck_v1.py` re-derived the cutoff gate
(`results/w132/stronger_model_gate/gate_recheck_v1.json`): verdict
`NO_CERTIFIABLE_STRONGER_MODEL`, **decision CID `258b6ed7…` invariant** (byte-identical
W114→W132), registry split {KNOWN:1, UNKNOWN:3 (+GLM-5 = 4)}, gate **CLOSED**. No new
primary cutoff disclosure since the W131 re-check earlier on 2026-06-02.

**The W132 γ contribution:** resistance-by-construction REMOVES the W131 cutoff-disclosure
dependency. The minted instances did not exist before the mint date, so the field is
resistant for ANY model regardless of disclosure — UNKNOWN-cutoff stronger models may be
used as DEV_ONLY characterization, while the FRONTIER claim stays on Maverick precisely
because its cutoff is KNOWN (Aug-2024) and the mint date strictly post-dates it. The
W123→W131 caps stay closed; no 405B run.

**graphify:** refreshed at START (HEAD `3708ea5d`; `results/w132/graphify/graphify_start_w132.txt`)
and END; the new `resistant_by_construction_battlefield_v1` creates the first semantic
bridge from a minted-task GENERATOR to BOTH the validated bench (`icpc_reflexion_bench_v1`,
`IcpcPilotProblemV1`) AND the audited grader (`coordpy_icpc_battlefield_v1`,
`grade_icpc_candidate_case_v1`).

---

## Net + carry-forward

* **W89 (+5.56) + W105 (+7.00) remain the only two confirmed retirements.** W132 retires
  none (no single-seed retirement; and the Maverick frontier pilot did not execute).
* **NEW `W132-T-RESISTANT-BY-CONSTRUCTION-BATTLEFIELD-MINTABLE`** — CoordPy CAN mint a
  genuinely-new, exact-oracle, post-cutoff, novelty-clean, deterministic resistant
  battlefield (33 ≥ 30) targeted at the failure families that beat the stack; this removes
  the W123 official-supply cap and the W131 cutoff-disclosure dependency as blockers on the
  EXISTENCE of a resistant instrument. The instrument STANDS as a reusable, push-button
  resistant battlefield.
* **`W132-L-MAVERICK-FRONTIER-PILOT-NIM-INFRA-BLOCKED`** — operational, not science; the
  earned pilot is push-button and re-runs when the NIM endpoint recovers.
* W123→W131 caps carried forward unchanged; decision CID `258b6ed7…` invariant.

**W133** (per RUNBOOK §12): run the earned Maverick frontier pilot when NIM recovers
(push-button), OR an operator-greenlit DEV_ONLY stronger-model characterization on the
minted field; a retirement is registered ONLY on a later clean multi-seed
`PASS_MECHANISM_DRIVEN`.

**No version bump (0.5.20 / coordpy.sdk.v3.43); no PyPI; `coordpy/__init__.py` untouched.**
