# RUNBOOK — W122: matched-family multi-seed closure on ICPC + same-family different-mechanism probe + stronger-model gate

**Status: LOCKED before any NIM call (2026-05-31).** Sibling of `COO-9`; executes the
W121 § 9 `CONFOUND_WEAKENS` branch's pre-committed optional tightening ("one paired seed
on BOTH battlefields to tighten the single-seed caveat") promoted to the **main empirical
lane**. `ultracode` OFF. Stable boundary: `coordpy.__version__ == "0.5.20"`,
`coordpy.SDK_VERSION == "coordpy.sdk.v3.43"`, no PyPI, `coordpy/__init__.py` untouched,
advanced work explicit-import only.

W120 built a ≥30 official-ICPC **RESISTANT** battlefield, certified Maverick, ran the
earned pilot ⇒ **B−A1 = +0.00 pp, FAIL** (seed 120001). W121 built the matched **EXPOSED**
control on the SAME official ICPC family, certified Maverick on the exposed side, ran the
earned pilot ⇒ **B−A1 = +3.33 pp, FAIL** (seed 120001) — both within the ±3.34 pp null
band ⇒ `CONFOUND_WEAKENS / bounded ceiling HARDENS`. The ONE remaining live statistical
caveat: **single seed each side.** W122 is **NOT** another battlefield, **NOT** another
"weakened/not proven" memo, and **NOT** a bounded-context/compaction/summarization job. It
is a **targeted closure operation** on that single-seed caveat + a same-family
different-mechanism probe + a stronger-model gate that only fires if it honestly opens.

This is **not seed-chasing.** It is the pre-committed removal of the one live caveat on the
matched-family contrast, run under a symmetric, decisive closure rule.

---

## 1. Exact α / β / γ branch logic

* **Lane α — matched-family multi-seed closure on ICPC (the main empirical lane).**
  * Reuse the EXACT W120 resistant 30-slice (CID `01bf9ef8…`) and the EXACT W121 exposed
    30-slice (CID `32d15db5…`). Do NOT drift the family, evaluator, cutoff rules, K, or
    budget shape. Both slices are CID-guarded in the drivers and reproduce byte-identically
    (verified NIM-free 2026-05-31: re-derived slice CIDs == provenance == pilot guard).
  * The existing seed on both fields (120001) is already present (W120 + W121). Run **one
    paired new seed (120002) on BOTH the exposed AND the resistant fields first** (the only
    move that can change the single-seed interpretation). Evaluate the **2-seed aggregate**
    under the § 2 closure rule. Run a **second paired new seed (120003) on BOTH fields ONLY
    IF the 2-seed aggregate is `AMBIGUOUS` (§ 2 B4).**
  * **Pilot ⟺** both slices reproduce (CID guards pass) ∧ Maverick stays pilot-admissible
    on both fields (Lane γ). On the live pass both hold ⇒ the paired-seed pilot is EARNED.
* **Lane β — same-family different-mechanism probe (mandatory; start NIM-free).** Audit
  whether the official-ICPC package family gives the strongest non-reflexion mechanism (M3,
  the executor-grounded structured-failure patcher, `coordpy.executor_grounded_patcher_v1`)
  a materially-useful signal that reflexion does NOT already have. Build + spend only if M3
  is genuinely different from plain reflexion on ICPC AND the signal is rich enough (§ 4).
  Otherwise KILL the lane honestly, NIM-free, with the machine-checkable reason.
* **Lane γ — stronger-model gate / graphify / truth (mandatory; NIM-free unless it opens).**
  Re-check primary-source cutoffs (§ 5). If a stronger-than-Maverick model becomes
  primary-KNOWN AND certifies on BOTH ICPC battlefields, it can supersede Maverick as the
  lead. If not, say so sharply and keep Maverick as the matched-family closure target.
  Refresh graphify at START + END (§ 7).

---

## 2. Exact paired-seed closure rule + ambiguity band (symmetric)

Constants (carried VERBATIM from W121 for continuity):

* `NULL_BAND_PP = 3.34` (one K=5 rescue at n=30 = 3.333 pp).
* `MARGIN_PASS_PP = 5.00` (W89/W105 retirement-grade margin).
* Seeds: existing `120001` (both fields); paired new `120002`; conditional 3rd `120003`.

Let `R̄` = mean RESISTANT B−A1 over the run seeds; `Ē` = mean EXPOSED B−A1 over the run
seeds (2-seed mean after 120002; 3-seed mean after 120003). The verdict is decided by the
following **pre-committed, exhaustive, precedence-ordered** branches (the SAME ±3.34 / +5.00
thresholds applied symmetrically to both fields):

1. **B3 `RESISTANT_SUPERIORITY_MULTISEED_CANDIDATE`** — `R̄ ≥ MARGIN_PASS_PP`
   (resistant rises to the margin, with or without exposed). The mechanism beats A1 on
   contamination-RESISTANT official ICPC code across multiple seeds ⇒ a candidate **THIRD
   retirement on resistant code** — the strongest possible pro-mechanism / anti-contamination
   result. Register SHARPLY. Do **NOT** declare a retirement on 2–3 seeds at n=30; W123 =
   the full resistant retirement bench (3+ seeds × ≥100, the W89/W105 bar). A clean
   candidate additionally requires BOTH resistant seeds individually `PASS_MECHANISM_DRIVEN`
   (9/9 ∧ MLB-1 ∧ MLB-2); a ≥5 pp mean WITHOUT clean per-seed gates is recorded as
   `MARGIN_WITHOUT_MECHANISM` and treated as **not** a clean rise (falls through to B4 if it
   does not also satisfy B1).
2. **B2 `EXPOSED_MARGIN_RESTRENGTHENS_CONTAMINATION`** — `Ē ≥ MARGIN_PASS_PP` AND
   `|R̄| ≤ NULL_BAND_PP` (exposed rises to the margin while resistant stays null). Exposure
   within ICPC reproduces the retirement-grade margin once de-noised ⇒ the W121 "contamination
   WEAKENED" read **REVISES BACK UP** toward contamination-supported; the difficulty/family-ease
   loophole closes via an exposed margin. Register SHARPLY. A clean call additionally requires
   BOTH exposed seeds individually `PASS_MECHANISM_DRIVEN`.
3. **B1 `MATCHED_FAMILY_NULL_SURVIVES_MULTISEED_CAVEAT_CLOSED`** — `|R̄| ≤ NULL_BAND_PP`
   AND `|Ē| ≤ NULL_BAND_PP` (both means stay in the null band). The matched-family null
   SURVIVES multiple seeds ⇒ the single-seed caveat is **CLOSED SHARPLY**; the
   contamination-confound stays WEAKENED and is now multi-seed; the bounded ceiling HARDENS
   to **multi-seed HumanEval-family-(ease/structure)-specific @ 70B**. (Expected outcome
   given seed 120001 + the Lane-β structural finding, but NOT assumed.)
4. **B4 `AMBIGUOUS_THIRD_PAIRED_SEED_EARNED`** — anything else: either mean lands in the
   open interval `(NULL_BAND_PP, MARGIN_PASS_PP)`, OR a mixed pattern (one field rises into
   the gap while the other is null/negative beyond the band), OR a ≥5 pp mean without clean
   per-seed gates that is not also a B1 null. ⇒ earn EXACTLY ONE more paired seed (120003)
   on BOTH fields and re-evaluate under this same rule. Do **NOT** buy a 4th.

**Decisiveness note.** At n=30, 2-seed means lie on the lattice {0, 1.67, 3.33, 5.00, 6.67,
…} (multiples of 1.667 pp). The values `3.33 ≤ 3.34` (null) and `5.00 ≥ 5.00` (margin) are
lattice points; the ambiguous open interval `(3.34, 5.00)` contains **no lattice point**.
So the 2-seed aggregate is structurally decisive on each field unless the two seeds disagree
in direction (the only realistic B4 trigger). **If the 2-seed aggregate is already decisive
(B1/B2/B3), do NOT buy the 3rd seed just because it is there.**

---

## 3. Exact same-family exposed-vs-resistant interpretation rule

The W121 contrast (`interpret_exposed_vs_resistant_v1`) compared single-seed margins. W122
generalises it to the multi-seed means via § 2 and records the **delta vs W121**:

* If B1 (both null, multiseed): the W121 single-seed `CONFOUND_WEAKENS` finding is
  **CONFIRMED + HARDENED** (now multi-seed). Contamination demoted from dominant driver to
  at-most-minor contributor, multi-seed. The honest residual (faint sub-floor
  exposure-consistent gradient) is re-measured on more seeds and reported either way.
* If B2 (exposed margin, resistant null): the W121 finding is **REVISED** — the exposed
  margin was real but single-seed-noisy at W121; de-noising reveals it ⇒ contamination
  reading RE-STRENGTHENS. This is a material reversal and must be reported as such, NOT
  buried.
* If B3 (resistant margin): the bounded **resistant** ceiling is OVERTURNED at multi-seed
  ⇒ resistant superiority is no longer 0-clean ⇒ candidate third retirement; the whole
  contamination framing is reopened (resistant code now shows the margin). Report as the
  headline.
* If B4: the contrast is genuinely ambiguous at 2 seeds; the 3rd seed decides; no
  interpretation is published until B1/B2/B3 resolves.

In ALL branches: difficulty comparability is re-checked empirically (per-seed A0 on each
field; exposed A0 must remain ≤ resistant A0 for the matched-difficulty design to hold), and
the same anti-cheat (reflexion sees only public samples + judge verdict bit + stderr; never
secret data) is invariant.

---

## 4. Exact M3 / different-mechanism earn-no-earn rule (Lane β)

**Pre-committed NIM-free audit (`audit_icpc_mechanism_signal_v1`).** M3's load-bearing
differentiator vs reflexion (W111 § 2.3) is the **explicit expected/actual contract
extracted from the failing test**. On BigCodeBench the hidden `unittest` oracle prints
`AssertionError: <actual> != <expected>` ⇒ M3 reads the expected value of the HIDDEN test.
On official ICPC the hidden oracle is **secret token-diff** that returns only "wrong answer
on a hidden case" — it NEVER reveals expected (anti-cheat). The ONLY expected-value signal
on ICPC is the **public samples**, which the existing reflexion bench ALREADY feeds
(`WRONG (expected …)`).

Audit classifies each failing reflexion turn (from the W120 + W121 sidecars) into
`{PUBLIC_SAMPLE_WRONG, HIDDEN_ONLY, RUNTIME_TRACEBACK, TIMEOUT, NO_SIGNAL}` and computes
`m3_exclusive_signal_fraction` = fraction of failing turns where M3 would hold an
expected/actual contract that reflexion does NOT already have. On ICPC this is the
`HIDDEN_ONLY` turns where the hidden expected is available — which is **0** (the hidden
expected is secret).

**Earn rule (pre-committed):**

* **BUILD + earn a probe ⟺** `m3_exclusive_signal_fraction ≥ M3_SIGNAL_FLOOR = 0.33`
  (the W111/W106 33 % floor: M3 must hold an exclusive actionable contract on ≥ 1/3 of
  failures to even plausibly clear reflexion, which already failed at 25 % rescue with the
  public-sample expected in the majority). On ICPC the fraction is ~0 ⇒ this branch does
  NOT fire.
* **KILL NIM-free ⟺** `m3_exclusive_signal_fraction < 0.33`. The ICPC secret-token-diff
  grading regime structurally denies M3 its differentiator ⇒ M3 reduces to a reflexion
  prompt-variant, is NOT genuinely different on this family, and (being strictly
  signal-poorer than the BigCodeBench setting where it was already sub-reflexion at 12.5 %)
  cannot earn a fair probe. **$0 NIM on Lane β.** This is the honest aggressive answer to
  "is the ICPC null reflexion-specific?": NO — the family's grading regime makes the
  ceiling mechanism-robust, not merely reflexion-specific.

The audit ships as executable code + a falsifiability test (a synthetic sidecar whose
hidden failures carry an M3-exclusive expected/actual contract ≥ 33 % DOES flip the verdict
to BUILD), so the kill is machine-checkable and reproducible, not prose.

---

## 5. Exact per-model disclosure-status + certification rule (Lane γ)

Reuse the W114 `certify_model_v1` / W121 `certify_model_exposed_v1` gates (C1∧C2∧C3∧C4 /
C1∧C2e∧C3∧C4) on BOTH battlefields. Disclosure matrix (primary sources, re-checked LIVE
2026-05-31):

| model | primary cutoff | status | reachable | certifiable on BOTH ICPC fields |
|---|---|---|---|---|
| `meta/llama-4-maverick-17b-128e-instruct` | **August 2024** (llama.com / Meta llama4 MODEL_CARD, re-confirmed) | **KNOWN** | ✅ | **YES** (the matched-family target) |
| `qwen/qwen3-coder-480b-a35b-instruct` | none stated (HF card + tech report) | UNKNOWN | ✅ | no (C1) |
| `deepseek-ai/deepseek-v4-pro` | none stated (V4 card; latest documented is V3 line) | UNKNOWN | ✅ | no (C1) |
| `mistralai/mistral-small-4-119b-2603` | none stated (Mistral docs) | UNKNOWN | ✅ | no (C1) |
| `zai-org/glm-5` | none primary (listicle only) | UNKNOWN | ❌ | no (C1 + reachability) |

`{KNOWN: 1, UNKNOWN: 4}`; **nothing newly primary-disclosed since W121** (re-verified live:
Maverick Aug-2024 KNOWN; DeepSeek-V4 / Qwen3-Coder-480B / Mistral-Small-4 / GLM-5 all
UNKNOWN; no new reachable stronger model with a primary-KNOWN cutoff). Decision CID
`258b6ed7…` invariant. **The stronger-model gate does NOT open ⇒ Maverick stays the
matched-family closure target.** If a stronger-than-Maverick model becomes primary-KNOWN
and certifies on BOTH battlefields in a future milestone, it supersedes Maverick.

---

## 6. Exact spend rules

* **Paired-seed spend (Lane α) is EARNED directly** — it is the main closure line. Seed
  120002 on BOTH fields = 2 × 30 × (A0 1 + A1 5 + B 5) = **660 NIM calls**, plus a 2-problem
  canary per field (~44 calls) to validate the harness end-to-end before the full 30.
* **3rd paired seed (120003)** = earned ONLY if the 2-seed aggregate is B4 `AMBIGUOUS`
  (§ 2). Another ~660 calls. Do NOT buy it if the 2-seed aggregate is decisive (B1/B2/B3).
* **Mechanism spend (Lane β) is NOT automatic** — earned only if `audit` returns BUILD
  (`m3_exclusive_signal_fraction ≥ 0.33`). On the live pass it returns KILL ⇒ **$0 NIM**.
* **Stronger-model spend (Lane γ) is NOT automatic** — earned only if a stronger model
  becomes primary-KNOWN + certifies on BOTH fields. On the live pass the gate is closed ⇒
  **$0 NIM**.
* Do NOT spend on extra seeds outside the pre-committed paired plan. Do NOT count a close
  edge as closure. No 405B run (unreachable; closed). No APPS main-lane NIM. No reopening
  MBPP+ V2 / frozen cross-modal / the closed Llama-3.1 rescue branch. No dirty exposed
  benchmark sold as a frontier win.

---

## 7. Exact graphify deliverables

* Refresh `graphify update .` at START (graph already live-current to the working tree;
  W121 symbols resolve in `graph.json`; the `GRAPH_REPORT.md` header read `8b15e976` because
  no topology had changed) and at END after the W122 module/scripts/tests land (force the
  report to rebuild to the W122 commit). Record the END commit.
* Use graphify for file selection + dependency checks:
  `graphify explain run_exposed_control_construction_v1 / run_icpc_public_construction_v1 /
  run_icpc_stdin_executor_v1 / certify_model_v1 / run_executor_grounded_patcher_bench_v1`;
  `graphify path run_exposed_control_construction_v1 run_icpc_public_construction_v1`;
  `graphify affected run_icpc_public_construction_v1`; `graphify query` as a secondary
  claim-surface finder. Confirm the W122 closure module's reuse edges (paired-seed closure
  → exposed control + battlefield bench + verbatim W108 gates; signal audit → sidecar
  classification).

---

## 8. Exact W123 branch logic

* **If B1 `…CAVEAT_CLOSED`:** the matched-family null is multi-seed-confirmed. W123 = accept
  the multi-seed-hardened bounded ceiling and pursue a GENUINELY DIFFERENT axis (NOT another
  ICPC seed/rerun); OR a stronger-than-Maverick primary-KNOWN model on BOTH battlefields if
  one opens.
* **If B2 `…EXPOSED…RESTRENGTHENS…`:** the contamination reading is back on the table. W123
  = a second matched exposure control on a DIFFERENT family to confirm the restrengthening
  (NOT a third ICPC seed).
* **If B3 `RESISTANT_SUPERIORITY_MULTISEED_CANDIDATE`:** W123 = the full resistant retirement
  bench (3+ seeds × ≥100 problems, the W89/W105 bar) to confirm a THIRD retirement on
  resistant code.
* **If B4 after the 3rd seed:** register the residual ambiguity; W123 = accept the bounded
  claim or escalate to a larger n PER FIELD (not more seeds at n=30).
* **Cross-cutting (any outcome):** a reachable stronger-than-Maverick model disclosing a
  primary-KNOWN cutoff re-opens BOTH matched battlefields with the strongest honest target;
  a further official ICPC surface widens either side.

`COO-9` stays the lead path unless the evidence forces a different code-line move. No version
bump; no PyPI; `coordpy/__init__.py` untouched.
