# RESULTS — W112 Lane β: NIM-free fair-reachability re-mining ⇒ no fair M3 strengthening can clear the 33 % floor (KILL at $0)

**Verdict: `NO_FAIR_STRENGTHENING_CAN_CLEAR_FLOOR_KILL_AT_0`. The fair in-budget
M3-strengthening design space is STRUCTURALLY capped at the 33 % MLB-2 floor by
BigCodeBench's hidden-test-coupling. The reliably fair-reachable ceiling is
8.3 %; even the best-CONCEIVABLE bound (33.3 %) only TOUCHES the floor with zero
headroom. No strengthened M3 variant earns live NIM.**

W112 Lane β is mandatory (`docs/RUNBOOK_W112.md` § 2): mine the resistant
hidden-test-coupling / semantic failure regime HARDER than W111, build only
FAIR strengthening ideas, kill weak ones at $0, and earn live NIM ONLY if a
strengthened M3 has a credible path ABOVE the 33 % floor. This note records the
$0 finding that **structurally strengthens** the W111 EMPIRICAL sub-floor result
(`W111-L-M3-PATCHER-SUB-REFLEXION-…`).

---

## 1. The harder mining (NIM-free; $0)

`scripts/mine_w112_fair_reachability_v1.py` re-executes the EXISTING W110
BigCodeBench pilot transcripts (`response_text` already on disk;
`results/w110/bigcodebench_pilot/…`) through the real deterministic `unittest`
executor (bcb_venv, headless Agg) and, for every problem in the **MLB-2
DENOMINATOR** (attempt-0 sample failed ⇒ the patch loop is invoked), classifies
the failure's **fair-reachability**: is the information to fix it present in the
FAIR regime (visible docstring + executor `stderr_tail`), or only in the hidden
`test` source? It spends ZERO model calls and NEVER feeds the test source to a
solver — the test is read only for OFFLINE difficulty characterisation (as the
W111 census did).

The fair-reachable fraction of the invoked set is a **STRUCTURAL CEILING on
MLB-2** for ANY fair same-budget patcher: an oracle-entangled failure sits
permanently in the denominator and can never be rescued without leakage.

**Artifact:** `results/w112/fair_reachability/w110_bcb_fair_reachability.json`.

---

## 2. Result — the MLB-2 denominator is dominated by oracle-entangled failures

MLB-2 denominator (attempt-0 failed / patch loop invoked) = **12 problems**:

| Reachability class | count | % of 12 | fair-reachable? |
|---|---|---|---|
| UNREACHABLE_MOCK_OR_FIXTURE | 4 | 33.3 % | NO — expected behaviour in the hidden test's mock/fixture |
| BORDERLINE_CONTRACT_UNDER_MOCK | 3 | 25.0 % | NOT RELIABLY — contract extracted but mock-entangled |
| UNREACHABLE_OTHER (ValueError, no contract) | 2 | 16.7 % | NO |
| **FAIR_REACHABLE_OUTPUT_VALUE** | **1** | **8.3 %** | YES — clean expected/actual contract, no mock |
| UNREACHABLE_TIMEOUT | 1 | 8.3 % | NO — non-termination, not digest-fixable |
| UNREACHABLE_UNDERSPEC_ASSERT | 1 | 8.3 % | NO — assertion fired, no extractable target |

* **STRICT (reliably fair-reachable) ceiling = 8.3 %** (1/12) — the single clean
  contract-no-mock problem. Far below the 33 % floor.
* **GENEROUS (best-conceivable) ceiling = 33.3 %** (4/12) — reached ONLY by
  counting all 3 `BORDERLINE_CONTRACT_UNDER_MOCK` problems as perfect rescues; it
  merely TOUCHES the floor (+0.3 pp headroom).
* **58 % (7/12) of the denominator is mock/fixture-coupled** — the expected
  behaviour lives in the hidden test, information-unavailable to any fair mechanism.

The mock-entangled "contracts" are exactly the W111 `/51` lateral-trade failure
pattern: a syntactic expected-value can be parsed from the stderr, but emitting
that value need not satisfy the hidden test's mock-INTERACTION assertions — so
they are NOT reliable rescues.

---

## 3. The four candidate strengthenings — all FAIR, all killed at $0

The earn rule (`docs/RUNBOOK_W112.md` § 2, conservative + falsifiable): a
strengthened M3 earns live NIM **iff the STRICT reliably-reachable ceiling ≥ the
33 % floor** (mock-entangled contracts are not counted as reliable rescues). No
strengthening can EXPAND the reliably-reachable set beyond the generous bound:

| ID | Idea (all FAIR: no test source, no oracle leakage, same visible-spec/stderr regime) | Marginal | Kill reason ($0) |
|---|---|---|---|
| **S-C** richer typed failure digest (assertAlmostEqual / assertTrue-with-locals / assertRaises / multi-line reprs / traceback frame localisation) | 4 invoked failures have assertion info the current parser missed | the newly-actionable failures are MOCK-COUPLED ⇒ raises the GENEROUS bound, NOT the reliably-reachable STRICT set |
| **S-A** multi-candidate failure aggregation (condition the patch on ALL K candidates + digests, not just the latest) | 3 invoked problems have ≥ 2 distinct A1 failure exception types | improves rescue EFFICIENCY on already-reachable problems; cannot expand the reachable set |
| **S-B** patch ranking / rejection (execute each patch; reject regressions vs prior best) | reduces the W111 lateral-trade losses (e.g. `/51`) | reduces REGRESSIONS only; adds NO new rescues ⇒ cannot raise the MLB-2 numerator |
| **S-D** visible-spec doctest invariants (parse `>>>` examples into local self-checks) | 12/12 invoked have a docstring | doctests are ALREADY in the visible prompt at generation time ⇒ a self-check adds ZERO new information (the failures are precisely where the hidden test demands MORE than the visible doctests) |

**Best-conceivable fair-strengthening MLB-2 upper bound = the generous ceiling =
33.3 %** — a perfect patcher rescuing every reachable AND every borderline
problem on the first patch. That merely touches the floor; every prior mechanism
lands well below it (W111 M3 12.5 %; W110 reflexion 25 %). The STRICT ceiling
(8.3 %) ≪ 33 % ⇒ **no fair strengthening has a credible path to CLEAR the floor.**

---

## 4. Honest interpretation

**What this IS — a STRUCTURAL strengthening of the W111 EMPIRICAL finding.** W111
measured ONE M3 variant sub-floor (MLB-2 12.5 %) on a rescue-concentrated probe.
W112 shows the *entire* fair in-budget M3-strengthening design space is capped at
the 33 % floor by BigCodeBench's hidden-test-coupling structure: 58 % of the
invoked-failure denominator is mock/fixture-coupled (oracle-entangled), only
8.3 % is reliably fair-reachable, and none of the four concrete strengthenings
(richer digest, multi-candidate aggregation, patch-rejection, doctest invariants)
can expand the reliably-reachable set. The 70B fair-mechanism search branch was
already closed by W111; W112 shows the close is structural, not one-variant-luck.

**What it is NOT.** It does NOT prove no mechanism can ever win on resistant code
— it shows the WITHIN-BUDGET, FAIR-REGIME (no test source) mechanism space at 70B
cannot, and the limit is the benchmark's oracle-coupling, not the digest design.
A genuinely STRONGER model (Lane α) is a separate axis. It does NOT retire
anything, weaken W89/W105, or prove the contamination confound.

**Answer to Lane β's mandated question:** *is there a fair strengthening of M3
worth future spend?* **No** — the fair design space is sub-floor by construction.
*Was W111 already enough to close the 70B mechanism-search branch?* **Yes**, and
W112 upgrades that close from empirical to structural.

---

## 5. Carry-forward

**Added:** `W112-T-FAIR-M3-STRENGTHENING-CEILING-SUB-FLOOR` — ESTABLISHED ($0
NIM). On the W110 BigCodeBench MLB-2 denominator (12 invoked), the reliably
fair-reachable ceiling is 8.3 % (1/12) and the best-conceivable bound 33.3 %
(4/12; 58 % mock/fixture-coupled); the four fair strengthenings (S-A..S-D) each
fail to expand the reliably-reachable set, so no fair M3 strengthening clears the
33 % floor. Structurally strengthens `W111-L-M3-PATCHER-SUB-REFLEXION-…`.

**Not retired:** the two confirmed retirements (W89, W105) — unchanged. Lane β
adds NO retirement and earns NO live NIM. The Lane β half of the bounded-claim
fallback condition (`docs/RUNBOOK_W112.md` § 6.2) is SATISFIED.
