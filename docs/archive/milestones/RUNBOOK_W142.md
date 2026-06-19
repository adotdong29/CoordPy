# RUNBOOK W142 — moderate-`p` family screen + high-K discover reliability + ≥3-family retirement attempt (only if honestly earned)

**Status: LOCK §1–§12 before any NIM spend. Builds on W141 ([[project_w141_milestone]]). `ultracode` OFF. Work on `main`. COO-9 lead.**

## 0. One-line thesis

W141 earned the programme's FIRST clean no-oracle contamination-resistant SAME-BUDGET superiority (`count_pairs_sum_le_t`, fair `p=0.08`, +70.6pp at K=4 over no-oracle verified-selection) but on exactly ONE family — short of the §7b ≥3-family / ≥2-mode retirement span. W141's own diagnosis: **the binding constraint is moderate-`p` family SUPPLY, not the verifier, not the mechanism.** The frontier 70B is BIMODAL per technique — it either KNOWS a technique (`p`≳0.7, verified-selection saturates at K≥4 ⇒ no headroom) or is CLUELESS (`p`≈0, no discoverable sample ⇒ nothing to extract). W142 attacks that one blocker directly: **(α) screen a wide resistant-by-construction family slate for moderate-`p` extractable supply under a FAIR neutral baseline; (β) make discovery reliable (raise discover-K, hold amortize-K) and, only if ≥3 honest moderate-`p` families exist, take the cheapest family-level retirement shot; (γ) primary-source research + architecture-requirements refinement + the stronger-model gate.**

W142 is NOT another anchor-only mechanism essay; NOT counting a one-family win as a retirement; NOT a supply hunt on official benchmarks; NOT a selector retry. It is a family-SUPPLY milestone.

## 1. Why the amortization win is multi-family by construction (the theory the screen must confirm)

At equal TOTAL family generation budget `G` over `M` same-technique members:
- **B0 (no-oracle verified-selection@K_a per member):** solves a member iff ≥1 of `K_a` plain draws is correct+efficient ⇒ `P_solve = 1 − (1−p)^{K_a}`. Total budget `M·K_a`.
- **ST (self-tutoring):** DISCOVER once (`K_d` draws on a teacher member ⇒ a no-oracle-verified winner ⇒ a self-derived holed-skeleton scaffold), then AMORTIZE (`K_a` SCAFFOLDED draws per member; scaffolded rate `q≈1.0` per W141). Total budget `K_d + M·K_a`.

At equal total budget (ST spends `K_d` on discovery + `G−K_d` amortized over members), ST's per-member superiority over B0 is **`(1−p)^{K_a}`** — the members B0 fails because it must RE-DISCOVER the rare technique on every problem while ST discovers once and transfers. This is retirement-grade (>+5pp) for `p ≲ 0.5` at `K_a=4` (`(1−p)^4`: p=0.10→66pp, 0.25→32pp, 0.40→13pp, 0.50→6.25pp) and **vanishes for high-`p`** families (`p=0.83→0.1pp`, the W141 NSL/inversions collapse) and for `p≈0` families (no discoverable sample). ⇒ the win lives EXACTLY in the moderate-`p` band, and it is INHERENTLY MULTI-FAMILY — it fires on every family with discoverable-but-rare supply. W142 tests whether ≥3 such families exist.

**Discover-vs-amortize accounting (locked).** The retirement claim is the EQUAL-TOTAL-BUDGET claim above (ST and B0 both spend `G` over the family; the per-member superiority is the amortization edge). Raising `K_d` (discover reliability) is paid out of `G` as a one-time family cost whose per-member share `K_d/M → 0` as `M` grows; it is reported SEPARATELY and never folded into a per-problem same-budget claim. `coordpy.discover_amortize_accounting_v1` computes and machine-checks the budget identity for every reported arm.

## 2. The fair neutral baseline (FNB) — the load-bearing methodology (locked, byte-stable)

The raw efficient-rate `p` MUST be measured under the TRULY-NEUTRAL self-consistency prompt — the exact W141-v4 baseline (`self_tutoring_controller_v1._efficient_prompt`):

> `{statement}\n\nWrite a Python 3 program that solves this problem. Read all input from stdin and write the answer to stdout in the exact format shown. Return ONLY one ```python code block.`

It names **NO technique**, gives **NO efficiency / time-limit / large-input / constraint-size cue**, and supplies **NO scaffold**. (W141 measured: naming the technique inflates `p` 0.32→0.92; even the words "largest input / time limit" inflate to 0.92. A cue hands B0 the discovery for free and MASKS the amortization win.) The statement itself must not name the data structure or imply the algorithm (enforced by G4 + the W138 parser-neutral surface). `p̂ = (passes on hidden bank) / K_screen`, graded SCORING-ONLY (a hidden-bank pass ⇒ correct AND efficient, since the naive TLEs/wrong-answers on hidden by construction).

## 3. Parser-neutrality / no-leakage rule (locked, enforced by code)

- **Parser-neutral I/O:** every family uses `IoShapeV1` + `render_normal_form_v1` via `make_pn_template`; `parser_neutrality_gate_v1` must PASS (no format that cues the algorithm). [G1]
- **No-leakage:** no official-task paraphrase; no accepted-solution reuse; no hidden-case reuse; no technique/efficiency cue in the FNB or statement; surface-disguised titles (the W138 discipline: "count pairs sum ≤ T" → "cheap pair tally"). A family is REJECTED if its apparent moderate-`p` could come from prompt leakage or format bias (G4 novelty + the FNB lock). [G4]
- **No-oracle audit (reused from W141 §2):** the mechanism's code path provably never reads the hidden bank / `ref_source` / `naive_source` / `brute_source` answer-key; `_passes_secret` is SCORING-ONLY. The S1 constraint-adversarial bank is model-self-generated in the audited subprocess. Machine-checkable.

## 4. The family-screen slate + the $0 admission gates (locked rule; slate fingerprinted before spend)

New module `coordpy.moderate_p_family_slate_v1`: **≥12 candidate family recipes** spanning the veins where the efficient technique is NON-OBVIOUS to a 70B (so it defaults to the O(N²) naive): counting-pair (sort+two-pointer / sort+sliding-window), counting-subarray (two-pointer / prefix-hash), monotonic-stack (harder than nearest-smaller, e.g. sum of subarray minimums), sweep / offline-ordering, BIT / Fenwick / prefix-order, binary-search-on-answer, and DP families ONLY if they admit leak-free accumulator-shaped extraction. Every recipe reuses the W138 `make_pn_template` discipline (exact-oracle `ref` / independent `brute` / admissible-wrong `naive` / parser-neutral shape / TIMEOUT or OUTPUT_MISMATCH discriminator).

New module `coordpy.moderate_p_family_screen_v1`: runs the **$0 gates FIRST** (reject before any NIM), then the NIM `p` measurement, then the admission rule. The slate fingerprint CID is LOCKED before any spend.

**$0 gates (no NIM):** G1 parser-neutrality; G2 exact-oracle discriminating (`ref` passes hidden, `naive` FAILS hidden, `brute`≡`ref` on a cross-check, independence); **G3 extractability** — `compile_tutor_from_winner_v1(ref_source, …)` returns a clean leak-passing tutor (substantive + completable + leak-clean holed skeleton); if the REF itself does not extract, a model winner of the same shape cannot ⇒ reject $0; G4 novelty / near-duplicate guard (distinct `algo_sig` + statement hash vs the existing slate and each other).

## 5. Moderate-`p` admission rule (locked BEFORE results)

A family is ADMITTED iff ALL of: G1 ∧ G2 ∧ G3 ∧ G4 pass ($0) **and** under the FNB on the frontier anchor `meta/llama-3.3-70b-instruct`:
- **G5 fair-`p` band:** `p̂ ∈ [0.10, 0.50]` **and** `wilson_interval_v1(passes, K_screen)` (95%) strictly excludes 0 and 1.
- **G6 discoverable supply:** at the locked discover budget `K_d`, `discover_self_scaffold_v1` on a teacher member yields ≥1 no-oracle-verified winner that compiles a scaffold (so DISCOVER does not always abstain).

`K_screen = 12` base; top up borderline families to ≤ 20 to tighten the Wilson interval. **Lane α SUCCEEDS iff: ≥3 admitted families, ≥30 admitted frontier-anchor instances total (members summed across families), and each admitted family has ≥ `M_min=4` members of supply for a family-level attempt.** If it cannot hit those thresholds honestly, land the screen anyway and register the exact supply blocker machine-checkably (the band is real but sparse / extraction-bound / cue-bound — a NAMED cap).

## 6. Budget accounting (locked, pre-registered)

- **Screen (Lane α):** per family `K_screen` FNB draws + 1 self-brute (the brute is only for the discoverable-supply check) graded on hidden. Staged: a `K=8` triage pass, then top up only `[0.10,0.50]`-candidates to `K_screen≤20`. Report exact call counts.
- **Pilot (Lane β, only if Lane α succeeds):** per admitted family — DISCOVER (`K_d + 2`: K_d candidates + 1 brute + 1 adversarial bank) + per member (`K_a` plain shared by A1/B0 + `K_re` scaffolded for ST + 1 brute + 1 bank). Arms A0/A1/B0/ST4/STd/STc/NEG (§7). The same-budget claim is the EQUAL-`G` claim of §1; `K_d` is reported as the amortized one-time cost with its per-member share `K_d/M`. `coordpy.discover_amortize_accounting_v1` emits the budget identity and FAILS the run if any arm's total exceeds its declared `G`.
- No exposed frontier-control spend by default; no new seed-chasing on old official benchmarks; no stronger-model frontier spend unless the §11 gate opens; Maverick cross-scale is OPTIONAL and separate from the main claim.

## 7. The mechanism slate (locked BEFORE any NIM) + earn rule

Arms (same hidden grader, pass/fail-only, the W105/W141 discipline):
- **A0** plain single-shot (1 draw).
- **A1** fair neutral self-consistency pool@`K_a` (the floor bar).
- **B0** no-oracle verified-selection@`K_a` per member (the STRONG bar — must re-discover per problem).
- **ST4** W141 self-tutoring at the original budget (`K_d = K_a`, the W141 shape).
- **STd** high-discover-K self-tutoring (`K_d` raised; amortize `K_a` held) — the W142 discover-reliability arm.
- **STc** ONE capability-aware controller/routing refinement (KEEP/APPLY by per-member discoverability) — included ONLY if honest and necessary; dominated arms dropped (the W140 lesson: always-apply scaffold beats a public-revert controller in the A1≈0 regime).
- **NEG** fake-scaffold negative control: a scaffold compiled from a verifier-REJECTED / deliberately-wrong sample must NOT lift the members (else the lift is scaffold-shape leakage, not technique transfer).

**Earn rule (retirement-grade, STRICT).** On the resistant frontier anchor, single seed then a confirming second seed:
1. on the LOCKED admitted family set, ST (ST4 or STd, pre-declared) beats **B0** by a retirement-grade margin (≥+5pp aggregated, at equal total family budget per §1);
2. the gain comes from transfer members B0 fails and ST solves (report per-member);
3. spans **≥3 admitted families** (the §7b span; the single-complexity-family exclusion does not bite ≥3);
4. `no_oracle_audit` PASS;
5. **NEG fails** (fake scaffold does not lift) and `fake_different` / diversity-real pass.
One or two families is NOT a retirement. A biased baseline is NOT a retirement. A discover-budget change reported as a per-problem same-budget claim is NOT a retirement. If ST beats A1 but not B0 ⇒ selection, not coordination ⇒ bounded claim. If the family set is < 3 ⇒ register the supply cap.

## 8. Self-test + regression-fixture rule (locked, before any NIM)

- **Build self-test ($0):** the slate self-test must show every $0 gate BITES — G1 fails a deliberately format-cueing input; G2 fails a non-discriminating (naive==ref) family; G3 fails a no-accumulator family; G4 fails a near-duplicate of `count_pairs_sum_le_t`. The admission filter is exercised end-to-end with a mock generator.
- **Regression fixtures (must pass before the pilot is trusted):** (a) the W141 `count_pairs_sum_le_t` win re-extracts cleanly (the extractor + leak gate are unchanged); (b) the W139 weak-tier safety (KEEP ≡ A1 non-negativity) holds for an un-extractable family; (c) the W138 anchor complexity wins' refs still extract. Run the existing W141 tests as a guard.

## 9. Frontier / retirement-target rule (locked)

Default frontier anchor = `meta/llama-3.3-70b-instruct` (the exact W105 retirement model; resistant by construction; primary-KNOWN cutoff ~Dec-2023). Maverick (`meta/llama-4-maverick-17b-128e-instruct`) = OPTIONAL push-button cross-scale check on the same admitted slice, separate from the main claim, only if its deployment is healthy. The main claim is the family-span retirement attempt, not a one-off anchor demo.

## 10. Primary-source research + architecture-requirements deliverable (Lane γ, mandatory)

Primary sources only (arXiv / OpenReview / official venue). Must answer: how to distinguish true family discoverability from prompt-induced cueing; how to measure `p` fairly; which remaining blockers are now ARCHITECTURE requirements rather than benchmark noise. Use the literature only if it changes the family screen, the mechanism, or the architecture requirements (no literature-summary-as-output). **Deliverable:** land `docs/ARCHITECTURE_REQUIREMENTS_W142_V4.md` (refining V3) — what the eventual coordination-native architecture must do to (R-new) DISCOVER low-`p` techniques reliably and AMORTIZE them across a family without oracle leakage (a discover-then-amortize head + a per-family supply/discoverability estimator). Re-check primary cutoff disclosures for Maverick / Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-119B-2603 / GLM-5 / any newly reachable stronger model; re-run the stronger-model gate recheck (decision CID `258b6ed7`).

## 11. graphify deliverables (locked)

- START: `graphify update .` from current HEAD (`4cb37ae`), confirm `graphify-out/GRAPH_REPORT.md` built-from == HEAD; `explain`/`path`/`affected` on the W141 cluster + band/calibration modules (done in recon).
- END: after the new modules + docs land, `graphify update .` again so `graphify-out/` matches repo truth; confirm the new screen/slate/accounting modules are real 1-hop bridges to the W141 controller + the W138 calibration machinery.

## 12. Carry-forward + W143 branch logic (locked)

- **Earned (≥3 families, ST>B0 retirement-grade, NEG fails):** register `W142-T-MODERATE-P-FAMILY-AMORTIZATION-IS-A-RESISTANT-COORDINATION-RETIREMENT` + a THIRD retirement; W143 = multi-seed + cross-scale (8B democratization + Maverick) + productionize the discover-amortize controller.
- **Capped (band real but < 3 families / extraction-bound / cue-bound):** register the EXACT supply cap (`W142-L-...`) machine-checkably; W143 target = the named blocker (a denser moderate-`p` generator, a code-competent local model with a different bimodal profile, or a primary-KNOWN stronger model when the `258b6ed7` gate opens). Keep W123–W141 caps closed unless new evidence genuinely changes them.
- W89 (+5.56) + W105 (+7.00) STAND as the only two retirements unless a clean retirement-grade earn is registered. No version bump (`0.5.20` / `coordpy.sdk.v3.43`); no PyPI; `coordpy/__init__.py` untouched; advanced work explicit-import only. Gate `258b6ed7` invariant.

## 13. De-risk synthesis (from the 3 $0 subagents) — LOCKED before spend

Three $0 subagents (family brainstorm / extractability analysis / primary-source literature) sharpen the plan. The four findings below are LOCKED into the screen design before any NIM:

1. **Extractability is the binding $0 gate, and it NARROWS the veins (decisive).** The W141 AST extractor only cleanly holes a SINGLE printed accumulator updated by `acc += <expr>` gated by either an enclosing `if` (pattern a) or an immediately-preceding shrink `while` (pattern b). Therefore: **sort+two-pointer counting (the W141 win) and sliding-window/two-deque counting EXTRACT CLEAN; prefix-hash (`seen[s]+=1` maintenance carries the technique, the `acc += seen[...]` has no gating predicate ⇒ the skeleton LEAKS) and binary-search-on-answer (the printed answer `lo` is a reassignment, not an accumulator; feasibility lives in an un-printed helper) STRUCTURALLY FAIL.** G3 runs `compile_tutor_from_winner_v1(ref_source, …)` on each family's canonical reference at $0 and MUST reject the prefix-hash + BSoA veins before any spend (a machine-checkable prediction). New families are authored in the clean accumulator-gated form (e.g. product-pairs as a both-ends two-pointer `if a[i]*a[j]<=T: cnt += j-i`).
2. **The viable moderate-`p` ∧ extractable veins are: (i) sort+two-pointer/sliding-window COUNTING (count_pairs_sum/absdiff/product, count_triples_sum, counting-subarray) — MODE_COMPLEXITY_BLIND; (ii) two-deque sliding-window (subarrays_sum_and_range, longest_subarray_absdiff_le_limit) — MODE_HIDDEN_EDGE; (iii) monotonic-stack (risky extraction).** The strongest §7b span is therefore the **≥2-MODE route**: ≥1 COMPLEXITY counting family + ≥1 HIDDEN_EDGE two-deque family — cleaner than 3 near-cousin counting surfaces. Aim for BOTH ≥3 distinct family names AND ≥2 modes; report the technique-CLASS count alongside the family-NAME count (do not sell 3 surfaces of one meta-technique as 3 independent families). **W140 directly measured the frontier 70B at A1=25% on the two-deque family** — the single best empirical prior that an extractable moderate-`p` family exists outside the counting-pair vein.
3. **Fair-`p` is measured over a small NEUTRAL-PROMPT BANK, not one phrasing (FormatSpread arXiv:2310.11324).** `p` is reported as the median over ≥2 truly-neutral phrasings (none naming a technique, efficiency, time-limit, largest-input, or data structure) + the spread as a robustness check; the band screen uses the median. The IRT peak-Fisher-information framing (metabench arXiv:2407.12844 / Fluid arXiv:2509.11106) is the principled justification for `[0.10,0.50]`: discrimination is maximal at `p≈0.5` and zero at the bimodal extremes — the band screen RANKS by closeness to 0.5 (reusing `headroom_band_calibration_v2`).
4. **Novelty + verifier are literature-defensible.** Closest amortization prior art = DreamCoder (arXiv:2006.08381) / Voyager (arXiv:2305.16291) (library/skill learning, but with oracle specs); RLAD (arXiv:2510.02263) is a CONTRAST (per-problem + oracle-warm-started), NOT precedent ⇒ W142's white space = **discover-once-amortize-many ∧ no-oracle on a frontier model**. The brute-on-small-inputs verifier (computes outputs, never predicts them) is the documented mitigation for the self-test false-positive failure mode (Self-Debug arXiv:2501.12793; ReST-EM arXiv:2312.06585) — keep the 0-FP/FN audit as a gate. Stronger-model gate re-confirmed CLOSED `{KNOWN:1, UNKNOWN:4}` against primary sources (Maverick Aug-2024 SETTLED; Qwen3-Coder-480B / DeepSeek-V4-pro / Mistral-Small-4-2603 / GLM-5 officially UNDISCLOSED; the aggregator "Mistral Nov-2024" is non-primary, REJECTED).

**Honest prior (locked).** The bimodal-generation wall is a real structural prediction; the moderate-`p` ∧ extractable ∧ span-distinct intersection may be THIN. A `<3`-family screen result is a real, publishable outcome (the mechanism stands as a de-risked first; the retirement is gated by the model's intrinsic bimodality, not by the coordination machinery). W142 commits to the honest result either way.

---
*Sections §1–§13 LOCKED (byte-stable) as of the W142 build, before any NIM spend. The FNB (§2 + §13.3 bank), the $0 gates (§4 + §13.1 extractability), the moderate-`p` admission rule (§5), the budget accounting (§6), and the earn rule (§7 + §13.2 mode-route span) are fixed before results.*
