# RESULTS W137 — parser-neutral HARD battlefield v2 + model-ladder hardness calibration

Pre-committed contract: `docs/RUNBOOK_W137.md` (LOCKED before any NIM). `ultracode` OFF. `COO-9`/`COO-62`.

> This is a living results doc. Lane α ($0 build + self-tests) is final. Lane β (mechanism bench) and
> Lane γ (frontier gate) sections are filled from the executed runs.

## 0. What W137 is (and is not)

W136 root-caused the W132–W135 "wrong-algorithm ceiling" as an **I/O-FORMAT CONFOUND** (the 70B model
wrote correct algorithms but misparsed the whitespace-flattened input) and showed the field is
one-shot / low-algorithm-headroom once I/O is normalised. W137 **rebuilds the generated field
parser-neutral and hardness-calibrated across a model ladder**, then re-tests the surviving
mechanisms only if the repaired field earns it. It is NOT another witness variant on the old
confounded field, NOT a broad rerun of the old W132–W135 slices, and NOT an official-benchmark
supply hunt.

## 1. Lane α — battlefield repair + hardness calibration ($0 build, SUCCESS)

### 1.1 The parser-neutral I/O kernel (the repair)

`coordpy.parser_neutral_io_v1` defines a canonical **one-logical-item-per-line normal form**
(`IoShapeV1` + `render_normal_form_v1`) and the **HC1 parser-neutrality gate**
(`parser_neutrality_gate_v1`): a minted case is parser-neutral iff a STRICT per-line reader (the
parser a model writes from the statement) and a read-all-tokens reader recover **byte-identical
structured data**, and the input is already in canonical normal form. The W132 confound — `_case`
flattening the whole body onto one line — is machine-impossible to recur: a flattened input FAILS
HC1 (the per-line reader raises / disagrees), the normal-form input PASSES. **Confirmed by the
confound-regression fixture: NF input → HC1 pass; W132-flattened knapsack input → HC1 fail.**

### 1.2 The hard slate (the headroom thesis)

`coordpy.hard_battlefield_slate_v2` — **17 templates, `slate_fingerprint_cid = 2ce207c5…`**, spanning
all four atlas modes, every input rendered in normal form. Per Lane-γ primary sources
(LiveCodeBench-Pro arXiv:2506.11928; AlgBench arXiv:2601.04996; competitive-error-analysis
arXiv:2506.22954), the slate targets the axis where 70B headroom lives — the NAIVE is
implementation-trivial but wrong by COMPLEXITY (TLEs the large hidden case) or by an unhandled CASE:

| mode | n | templates |
|---|---|---|
| COMPLEXITY_BLIND | 7 | count_inversions, longest_bounded_subarray, count_pairs_sum_le_T, count_pairs_absdiff_le_D, sum_nearest_smaller_left, count_light_blocks, max_j−i (A[i]≤A[j]) |
| WRONG_ALGORITHM_ADMISSIBLE | 5 | house_robber_circular, min_coins_arbitrary, weighted_interval_scheduling, knapsack_01, min_partition_diff |
| HIDDEN_EDGE_STATE_MISS | 2 | max_subarray (all-negative Kadane), longest_subarray_sum_K (first vs last prefix-index) |
| SEARCH_ENUM | 3 | climb_stairs_{1,2,3}, lattice_paths_blocked (parser-neutral grid), compositions_{1,2,3,4} |

The W132 knapsack, weighted-interval, and lattice traps are reborn here **parser-neutral** (the very
families W136 found confounded).

### 1.3 The $0 quality gates (all PASS)

- **HC1 parser-neutrality** + **HC2 exact-oracle discrimination**: **17/17** templates admitted
  (ref-solvable, INDEPENDENT brute == ref small-case agreement, naive looks-right-on-public /
  fails-hidden with the declared kind, split integrity) AND parser-neutral.
- **HC5 template diversity**: max pairwise statement Jaccard **0.4737 < 0.55**, all 17 distinct, 4
  modes (not one family repeated).
- **Deterministic regeneration + mint timeout-invariance**: same content-CID on re-mint and at mint
  timeout 3.0s vs 8.0s (the answer key is the reference's output; the naive TLEs at both).
- `scripts/run_w137_build_v2_battlefield_and_selftest_v1.py` → ALL SELFTESTS PASS;
  `results/w137/w137_build_selftest_v1.json`.

### 1.4 Model-ladder hardness calibration (HC3 + HC4)

`coordpy.model_ladder_calibration_v1` runs single-shot A0 across the locked ladder (small
`meta/llama-3.1-8b-instruct` + strong anchor `meta/llama-3.3-70b-instruct`) + a small strong-anchor
A1, then admits the **headroom band**: HC3 rejects a template the strong anchor one-shots (A0 ≥ 0.80);
HC4 rejects a template the strong anchor passes nothing on even with K samples. Per tinyBenchmarks
(arXiv:2402.14992) / Lost-in-Benchmarks (arXiv:2505.15055), discrimination — not raw difficulty — is
the selection criterion.

`scripts/run_w137_model_ladder_calibration_v1.py` → `results/w137/w137_calibration_v1.json`.

**CALIBRATION RESULT (ladder small `llama-3.1-8b` + strong anchor `llama-3.3-70b`; 3 A0 + 1 A1(K=3)
instances/template; 102 A0 gens + A1; `calibration_cid 8b0f6231…`; wall 2735s on a degraded NIM
endpoint).** The decisive, clean finding — at 70B the repaired field's difficulty is **BIMODAL**: every
template's strong-anchor pass@5 (A1) is **exactly 1.00 or 0.00 — nothing in between**.

| verdict | n | templates |
|---|---|---|
| **HC3-SATURATED** (strong A0 = 1.00; one-shot) | **12** | all 5 WRONG_ALGORITHM (knapsack, weighted-interval, circular-robber, min-coins, min-subset-diff), both HIDDEN_EDGE (all-negative Kadane, longest-subarray-sum-K), all 3 SEARCH_ENUM (stairs, lattice, compositions), 2 COMPLEXITY (count_inversions, longest_bounded_subarray) |
| **HC4-DEAD** (strong A0 = A1 = 0.00) | **4** | the harder COMPLEXITY (count_pairs_absdiff = sort+sliding-window, sum_nearest_smaller = monotonic-stack, count_subarrays = two-pointer, max_j−i = prefix/suffix) |
| **ADMITTED headroom band** (A0 < 0.80, best > 0) | **1** | `cb_count_pairs_sum_le_t` (A0 = 0.00, **A1 = 1.00**) |

**Surviving (headroom band): 1/17 ⇒ THRESHOLD MISS** (1 ≪ the 40/40/40/30 floors). Two readings,
both load-bearing:

1. **The 12 saturated templates GENERALIZE W136**: once I/O is parser-neutral, the textbook
   DP/greedy/counting traps do NOT fool the 70B's own generation — it one-shots them. The W132–W135
   "wrong-algorithm ceiling" was the I/O confound; with clean I/O there is no ceiling on these.
2. **The 4 dead templates** are the genuinely-hard non-obvious-efficient-algorithm problems the 70B
   cannot write even with K samples (A1 = 0) — the W128–W131 generation ceiling. The lone survivor
   has **A1 = 1.00**, so even it has no room for a mechanism to beat A1.

⇒ **No template has 0 < A1 < 1** — the only regime where a same-budget mechanism can beat A1. The
parser-neutral, hardness-calibrated field provides NO mechanism-sensitive multi-mode headroom at 70B.
This is the honest culmination of the W120→W136 arc, and it explains the 0-clean-resistant-superiority
result: at 70B, clean algorithmic difficulty is bimodal (one-shot or impossible), whereas the W89/W105
retirements are on HumanEval(+), which DOES have a middle band the 70B partially solves and reflexion
completes.

## 2. Lane β — repaired-field model-ladder mechanism bench

Same-budget arms (K=5; attempt-0 standard prompt; one model call/attempt; no early stop; graded on
secret; witness generation $0 outside K): A0 / A1 / B0 (blind reflexion) / C0 (exact-oracle complexity
witness) / **M1 = the auto-routing counterexample-else-complexity controller (LEAD)** / M2 (oracle-free
deployable) / M3 ($0 fake-different negative control). The fake-different discipline bites: the
relabeled-reflexion control M3 and B0 classify `FAKE_DIFFERENT`; C0/M1/M2 classify `REAL`.

`scripts/run_w137_mechanism_bench_v1.py` → `results/w137/<mode>/…/report.json`.

**DEV BENCH RESULT (bounded complexity-mechanism diagnostic).** Since calibration admitted only 1
template (THRESHOLD MISS), the formal §7a dev gate has no admitted multi-mode corpus. To close the one
inference gap calibration leaves — *can the complexity witness (the W133 mechanism) rescue the HC4-dead
complexity templates, a band A0/A1 cannot see?* — a bounded diagnostic ran A0/A1/B0/**M1** (the
auto-routing counterexample-else-complexity controller) on **3 complexity problems** (the 1 survivor +
2 HC4-dead), `meta/llama-3.3-70b-instruct`, K=5, 972s (`results/w137/dev/…/report.json`):

| arm | acc | vs A1 | vs B0 | note |
|---|---|---|---|---|
| A0 | 0.00 | — | — | single-shot fails all 3 |
| A1 | 33.33% | — | — | solves the survivor via sampling |
| B0 | 33.33% | +0.00 | — | blind reflexion rescues nothing extra |
| **M1** | **66.67%** | **+33.33** | **+33.33** | **rescues the dead `cb_count_pairs_absdiff`** (A1=B0=0 → pass) |

M1's MLB-1 = 0.667, MLB-2 = 0.50. **The exact-oracle complexity witness is REAL on the parser-neutral
field**: it got the 70B to write the efficient sort+sliding-window the i.i.d. samples + the blind
reject bit both missed — reconfirming W133 on a CLEAN (non-confounded) field. It did NOT crack
`cb_max_j_minus_i` (prefix-min/suffix-max two-pointer too non-obvious even with the witness).

**§7a verdict: FAIL by SPAN** (`modes=1<2, families=1<3`): the gain is COMPLEXITY-mode only, and every
non-complexity mode is one-shot-saturated (HC3), so the ≥2-mode span is **structurally unmeetable** —
exactly the W133 single-mode cap, reconfirmed on the clean field. The fake-different discipline bites:
M3 (relabeled reflexion) + B0 classify `FAKE_DIFFERENT`; C0/M1/M2 classify `REAL`. *Caveat: n=3,
1 rescue — a small-N diagnostic confirming the calibration's structural finding, NOT a robust gain.*
⇒ **NOT earned ⇒ $0 frontier.**

## 3. Lane γ — primary-source research + frontier gate

Primary-source findings that shaped the build (executable, not literature-summary): (1) re-target the
field onto the complexity/case-analysis axis (LiveCodeBench-Pro 2506.11928; AlgBench 2601.04996;
2506.22954); (2) I/O-format invariance as a first-class gate (FormatSpread 2310.11324; 2411.10541;
ReCode 2212.10264) ⇒ HC1; (3) construction-based resistance (DyCodeEval 2503.04149; GSM-Symbolic
2410.05229) strictly stronger than date-based ⇒ sidesteps the cutoff gate; (4) IRT discrimination
across a model ladder (tinyBenchmarks 2402.14992; 2505.15055) ⇒ HC3/HC4; (5) the rStar-Coder
mutual-verification + AutoCode VGC oracle recipe (2505.21297; 2510.12803) ⇒ the ref/brute/naive
discipline.

**Stronger-model cutoff gate re-derived in code:** `decide_certification_v1()` →
`NO_CERTIFIABLE_STRONGER_MODEL`, decision CID **`258b6ed794b45a18…`** (byte-identical to the W114
lock), `{KNOWN:1, UNKNOWN:…}`. Llama-4-Maverick (Aug-2024 KNOWN) is C1∧C2∧C3 but C4-settled (W113);
Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4 / GLM-5 are primary-UNDISCLOSED. **No new
certifiable stronger model.** Default frontier target stays `meta/llama-3.3-70b-instruct`.

**FRONTIER OUTCOME: NOT EARNED ⇒ $0 frontier.** Lane α missed the corpus thresholds (1/17) and the
§7a dev gate fails by SPAN, so §7b is not reached. No frontier rerun, no Maverick cross-scale spend.
The locked frontier target would be `meta/llama-3.3-70b-instruct`; it is not run.

## 4. Net

**At 70B, clean algorithmic difficulty is BIMODAL (one-shot or impossible), leaving no
mechanism-sensitive multi-mode headroom band.** This is the structural reason resistant code
superiority has been 0-clean across the entire W120→W136 arc, and it is the honest culmination of that
arc: the W132–W135 "ceiling" was an I/O confound (W136); once the field is parser-neutral AND
hardness-calibrated, the 70B either one-shots a template (12/17) or cannot do it even with K samples
(4/17), with essentially nothing in between (1/17). The exact-oracle complexity witness remains REAL
(reconfirmed on the clean field) but SINGLE-MODE, so it cannot clear the ≥2-mode §7a span when the
other modes are saturated. By contrast the **W89 (+5.56) + W105 (+7.00) retirements stand precisely
because HumanEval(+) HAS a middle band** the 70B partially solves and reflexion completes.

- **Retirements:** W89 + W105 remain the ONLY two; W137 retires none, earns no frontier, demonstrates
  no coordination superiority.
- **Carry-forward caps STAND** (W123→W136); + `W137-L-REPAIRED-FIELD-NO-MECHANISM-HEADROOM-BAND-AT-70B`
  (threshold MISS; §7a span structurally unmeetable) and
  `W137-T-COMPLEXITY-WITNESS-IS-REAL-BUT-SINGLE-MODE-ON-PARSER-NEUTRAL-FIELD`.
- **Assets landed (reusable, push-button):** the parser-neutral I/O kernel (the durable W136 fix), the
  hard slate v2, the model-ladder calibration harness, the corpus assembler, the mechanism bench.
- **W138** = a code-competent model with a populated 0 < A1 < 1 band (better generation AND
  verification) / a genuinely different (non-bimodal) battlefield axis / accept the bimodal-ceiling cap.
- **`COO-9` stays the lead path.** Stronger-model gate CLOSED (`258b6ed7`).

No version bump (`0.5.20` / `coordpy.sdk.v3.43`); no PyPI; `coordpy/__init__.py` untouched.
