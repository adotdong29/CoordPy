# RUNBOOK W141 — emergent family-level SELF-TUTORING with cross-problem transfer (no-oracle) → resistant coordination retirement attempt

**Status: DRAFT (lock §1–§10 before any NIM spend). Builds on W140 ([[project_w140_milestone]]). `ultracode` OFF. Work on `main`. COO-9 lead; sibling of COO-65.**

## 0. One-line thesis

W140 proved a teacher-compiled holed-skeleton TUTOR lifts the frontier 70B (`meta/llama-3.3-70b-instruct`) on resistant algorithmic families by +75–100pp on the techniques it lacks — but it used the **exact-oracle witness**, so it is *teaching*, not *coordination*. W141 removes the oracle: the team derives the family-level scaffold **from the problems it can already solve** and transfers it to the problems it cannot — a genuine multi-agent same-budget superiority on the contamination-resistant field, with **zero domain-knowledge injection**. If it beats same-budget self-consistency **and** no-oracle verified-selection on the frontier, it is the programme's **first resistant retirement**.

## 1. Why this can be a real advance (and not another cap)

The earn must clear the STRONGER no-oracle baseline, not just majority vote:
- **Baseline A — self-consistency (majority/first-of-K).** Weak on these families: the majority sample is the O(N²) brute (TLEs), so majority-vote selected accuracy is low.
- **Baseline B — verified-selection@K (W129-style, no oracle).** Pick the sample passing a self-written-brute correctness check (small self-gen inputs) + a complexity/timeout check (large self-gen input). On a problem with frontier A1∈(0,1), B finds the correct sample with prob 1−(1−A1)^K — strong when A1 is moderate. **This is the bar to beat.**
- **W141 novel value = FAMILY-LEVEL AMORTIZATION (the clean, defensible win).** Verified-selection must RE-DISCOVER the technique on EVERY problem — it pays the full per-problem discovery cost (K raw samples each) and is permanently capped at the per-problem discovery rate. W141 discovers the family technique ONCE (a few members of a same-technique family, e.g. `subarrays@N` across many N — the two-deque is the SAME technique at every N, only the brute's TLE threshold moves), compiles ONE family scaffold, and AMORTIZES it: applies a cheap scaffolded re-attempt to ALL remaining members. **At equal total generation budget over a family of M same-technique problems, discover-once-apply-many solves strictly more than discover-per-problem** — because discovery is the bottleneck and selection cannot amortize it. W140 PROVED the family scaffold transfers across difficulty (max_j 0→100, subarrays 25→100 under one scaffold); W141 shows the team can SELF-DERIVE that scaffold from the members it solves and reap the amortization. Worked estimate (p_raw≈0.25 two-deque rate, q_scaffold≈1.0, M=20, budget=80): verified-selection@K=4 ≈ 14/20; self-tutoring (discover with ~16 samples then ~1–3 scaffolded each) ≈ 19–20/20. The gap is the amortization advantage, and it WIDENS with M.
- **Secondary value = approach-extraction transfer** (riskier, only if the de-risk supports it): on a true A1≈0 member the right APPROACH may appear in a buggy sample more often than full correctness; if the no-oracle verifier can rank it up, the scaffold extracts the approach and re-gen fixes the bug. Treat as a STRETCH; the amortization win above does NOT depend on it.

**The honest retirement-vs-lift risk (from the cap survey).** A *lift* (beating self-consistency) is very likely; a *retirement* needs same-budget multi-agent superiority on the FRONTIER 70B clearing the §7b span (≥2 modes OR ≥3 families) with a clean margin AND `fake_different`/`fake_diversity` passing. The cap survey's caution: the 70B field is near-bimodal off the complexity mode, so a witness-style gain stays anchor-concentrated. **Why amortization answers this:** amortization is inherently MULTI-FAMILY — it works on EVERY family with frontier A1∈(0,1), and `headroom_band_slate_v3` ships 7 COMPLEXITY families. A complexity-mode earn across **≥3 distinct complexity families** clears the §7b ≥3-family span (the single-complexity-family exclusion does not apply to ≥3). So the W141 retirement path = amortized self-tutoring beating verified-selection on ≥3 complexity families at the frontier; chase ≥1 FUNC-mode family (`subarrays_sum_and_range`, `mod_then_maxsub`) for the cleaner 2-mode span as a bonus. If only 1–2 families earn ⇒ a W140-class resistant LIFT (still a programme first: the first NO-ORACLE resistant solve, which W125–W130 never achieved), registered as a bounded claim, NOT a retirement.

This **evades the prior caps** (to be confirmed by the cap-survey subagent): W125 re-routing (W141 generates NEW trajectories with the scaffold, not a re-pick of A1∪B); W128 selection (W141 LIFTS generation via the transferred scaffold, not just selects); W126/W127 (W141 operates where the technique IS discoverable on ≥1 family member, unlike the 0-ceiling uniformly-unsolved set). The risk it could still HIT: if the self-derived scaffold is materially weaker than the oracle scaffold (no-oracle extraction loses the technique), or if no resistant family has BOTH a discoverable member AND an A1=0 transfer target.

## 2. The NO-ORACLE GUARD (the load-bearing discipline — this is what makes it coordination, not teaching)

The mechanism MAY use, during the solve loop, ONLY:
- the PUBLIC problem statement + its public example I/O (given);
- SELF-GENERATED inputs drawn from the public constraints (the team writes a generator; constraints are public);
- a SELF-WRITTEN BRUTE — the obvious algorithm, a model generation — used as a slow-but-correct **correctness reference** on small self-gen inputs, after the brute is itself validated against the public examples;
- a COMPLEXITY/TIMEOUT check (run a candidate on a large self-gen input; TLE ⇒ reject) as an **efficiency** signal;
- CROSS-SAMPLE CONSENSUS on self-gen inputs.

The mechanism MUST NOT, anywhere in the loop, touch: the HIDDEN test bank (used ONLY for final scoring, never inside the mechanism), the exact-oracle witness / ref solution (W140 used it; W141 may NOT), or any external technique hint. **A machine-checkable `no_oracle_audit_v1` gate must pass: the mechanism's code path provably never reads the hidden bank or the ref.** This is the analogue of W140's leak gate and is non-negotiable.

## 3. The mechanism (concrete pipeline — locked from design recon)

NEW: `coordpy.self_tutoring_technique_extractor_v1` (the novelty) + `coordpy.self_tutoring_controller_v1` (orchestration + earn gate). Everything else composes from audited parts:

| step | action | function — module |
|---|---|---|
| 0 mint | resistant instance + grader | `mint_problem_v1(template.minted,…).to_pilot_problem(…)` — `resistant_by_construction_battlefield_v1`; families `CX_FACTORIES`/`FUNC_FACTORIES` — `headroom_band_slate_v3` |
| 1 generate K diverse | 1 ANALYZE + (K-1) IMPLEMENT, same-budget; emits sketches + self-gen counterexamples | `run_role_diverse_search_v1(gen, pilot, K)` — `role_diverse_algorithm_search_v1` (W128) |
| 2a no-oracle correctness | differential vs peers + self-written brute on self-gen SMALL inputs | `grade_candidates_v1` + `derive_auto_cases_v1` — `public_signal_selection_oracle_v1` (W129) |
| 2b no-oracle efficiency | measure the candidate's OWN runtime-growth + constraint-budget; ABSTAIN on low confidence (breaks the pawnshop tie, oracle-free) | **`deployable_complexity_witness_v1`** (W134): DW1 constraint-budget + DW2 self-runtime stress-growth ladder + log-log fit (R²>0.7 gate) — consumes only (code, statement, public samples); fall back to `derive_stress_cases_v1` (W129) |
| 2c pick winner | commit correct+efficient, ABSTAIN on tie | `select_so_v1(…, variant="SOLEAD")` → `.committed_code` — W129 |
| 3 extract scaffold (NEW) | AST-derive holes from winner's program → leak-cleared `FamilyTutorV1` | `compile_tutor_from_winner_v1(winner_code, template, problem)` (NEW) → validate via `tutor_leak_gate_v1` (+ new `discriminators=` param) + completability re-run |
| 4 amortize / re-attempt | inject scaffold across family members, KEEP/APPLY/revert | `run_tutor_split_arm_v1` / `run_tutor_controller_arm_v1` — `cross_tier_tutor_bench_v1` (W140) |
| base A1 | same-budget self-consistency (the floor bar) | `run_icpc_reflexion_bench_v1(…, K)` — `icpc_reflexion_bench_v1` |
| base B | no-oracle verified-selection@K per problem (the STRONG bar) | steps 1+2 per problem, no extraction/amortization |
| gate | earn rule + fake-different + diversity-real | `evaluate_gate_v1` + `fake_different_report_v1` + `DiversityReportV1.classify` |

**Non-negativity is structural:** the controller KEEP path makes every arm ≥ A1 regardless of extractor quality; a brittle/failed AST extraction is **discarded → KEEP ≡ A1** (never harms). The decisive no-oracle edge over the W128/W129 selection cap is step 2b (complexity/timeout), which separates the O(N²)-passes-public-but-TLEs candidate from the O(N) one — the exact `pawnshop` failure W129 localized.

## 4. Same-budget accounting

Total model generations for W141 over a family of M problems ≤ the budget given to Baseline B (verified-selection@K over the same M problems). Concretely fix a per-family generation budget G; W141 spends G as K_explore on a teacher subset + K_exploit on the transfer targets; Baseline B spends G as K per problem. The mechanism's verifier executions (brute/complexity runs) are LOCAL compute, not model calls — but report them. No hidden-test calls at all.

## 5. Preflight (NIM calibration — gates the pilot)

A family qualifies for the W141 transfer claim only if, at the frontier (3.3-70B), it has BOTH: (a) ≥1 member with A1∈[0.15, 0.7] (DISCOVERABLE — a correct sample appears, with headroom) AND (b) ≥1 member with A1≤0.10 (TRANSFER TARGET — not independently solved at budget). From W140: `subarrays_sum_and_range` is discoverable at @30000 (A1=25%); `max_j_minus_i_le` is an A1=0 target at @50000; tune the knob (N) on the W138 `headroom_band_slate_v3` families to populate ≥2 families with this mixed-difficulty profile. Calibrate A1 as a rate (n≥6) per member.

## 6. Earn rule (retirement-grade — STRICT)

On the resistant frontier (3.3-70B), single seed then a confirming second seed:
1. W141 selected accuracy − Baseline B (verified-selection) selected accuracy ≥ **+5pp**, at equal generation budget, aggregated over the family;
2. the gain must come from the TRANSFER TARGETS (A1≤0.10 members W141 solves that B does not) — report per-member;
3. spans ≥2 resistant families;
4. `no_oracle_audit_v1` PASS (no hidden/ref access in the loop);
5. the fake-scaffold negative control FAILS (below) — proving the verifier+extraction is load-bearing.
If all pass ⇒ a multi-agent same-budget superiority on the contamination-resistant field ⇒ candidate THIRD retirement (first resistant). If only Baseline A is beaten (not B) ⇒ NOT a retirement (selection, not coordination) ⇒ bounded claim. If transfer targets are not solved ⇒ cap.

## 7. Controls / falsifiers

- **no-oracle audit** (§2): provably no hidden/ref access.
- **fake-scaffold negative control**: a scaffold compiled from a sample the verifier RANKED WRONG (or from a deliberately-wrong program) must NOT lift the transfer targets — else the "lift" is scaffold-shape leakage, not technique transfer.
- **discovery-bound control**: on a family where EVERY member has A1=0 (nothing discoverable), W141 must FAIL (it cannot invent a technique no agent found) — confirms it is honest transfer, not magic, and locates the boundary (this is the W126/W127 cap, expected).
- **budget audit**: W141 generation count ≤ Baseline B count.
- **fake-different / fake-diversity** checks reused from `cross_tier_tutor_bench_v1` / W128.

## 8. Stronger-model + version discipline

Gate `258b6ed7` invariant (re-run the W140 recheck script); the resistant field needs no stronger model (resistant by construction). No version bump (`0.5.20` / `coordpy.sdk.v3.43`); no PyPI; `coordpy/__init__.py` untouched; all advanced work explicit-import-only. Frontier = `meta/llama-3.3-70b-instruct`. NIM endpoint was UNSTABLE in W140 (oscillating 4s↔36s + HTTP-500) — budget extra wall-time, retry, and concurrency.

## 9. Carry-forward on close

If earned: `W141-T-EMERGENT-SELF-TUTORING-TRANSFER-IS-A-RESISTANT-COORDINATION-SUPERIORITY` + a third retirement registered. If capped: a precise cap theorem (which of: no-oracle-extraction-too-weak / no-discoverable-member / transfer-fails-on-A1=0 / ties-verified-selection). W89+W105 STAND unless a clean retirement-grade earn is registered.

## 10. W142 pointer

Earned → multi-seed + cross-scale (8B) + productionize the self-tutoring controller. Capped → the precise blocker becomes W142's target.

## 11. De-risk verdict (from the 3 $0 subagents + the in-repo self-tests) — LOCKED

- **Verifier de-risked POSITIVE.** On 115 REAL 70B baseline samples (the W140 sidecar), the no-oracle S1∧S2 verdict is BYTE-IDENTICAL to the hidden-test grader: **0 false-positive, 0 false-negative** across 4 families. S1 (self-brute agreement) is STRICTLY load-bearing — it caught 8/8 real fast-but-WRONG candidates that S2-alone accepts (47% FP); the efficiency signal alone is unsafe. This is the empirical W125-cap escape.
- **S1 bank MUST be constraint-adversarial** (28/30 discrimination vs 0/30 public-only). Implemented: `_build_small_bank_v1` = template random + public mutations + a MODEL-written constraint-binding generator exec'd in the audited subprocess.
- **The binding constraint is efficient-sample SUPPLY, not the verifier.** At 70B: `sum_nearest_smaller_left` has genuine supply (9/28 monotonic-stack) — the clean target; `subarrays_sum_and_range` / `kth_smallest_pair_distance` / `max_j_minus_i_le` are ~0 supply (nothing to extract; the verifier correctly returns ∅, no false hope). **PRE-SCREEN supply: if `select_winner_v1` abstains, the family yields no scaffold ⇒ KEEP ≡ A1 (correct $0 abstain).**
- **$0 build validated:** the extractor produces a clean leak-passing holed skeleton on 5/9 families (the count/two-pointer/stack shapes; under-blanked ones are gate-discarded → KEEP); the verifier picks the correct+efficient program on the complexity-blind families via the efficiency signal; the full controller plumbing (discover→extract→amortize) runs end-to-end (mock gen). Non-negativity is structural (KEEP fallback).
- **The amortization earn fires in the LOW-per-problem-supply regime** (verified-selection@K often misses, but the once-discovered scaffold lifts re-gen): target families/knobs with frontier efficient-rate ~8–25%. NSL (~32%) is the de-risked anchor; screen `count_pairs_sum_le_t`, `count_pairs_absdiff_le_d`, `longest_bounded_subarray` for supply in the dev probe.

## 12. Earn discipline (NIM)

A graded resistant pilot is OPERATOR-GREENLIT only. The $0 build + a CHEAP dev probe (supply screen + a single-family discover→amortize on the de-risked NSL) come first; escalate to a multi-family resistant pilot only if the dev probe shows ST > verified-selection at equal budget with the no-oracle audit passing. NIM endpoint was UNSTABLE in W140 — budget wall-time/retries.

---
*Sections §1–§12 LOCKED (byte-stable) as of the W141 build. The mechanism is $0-validated; the no-oracle guard (§2) + supply pre-screen + KEEP fallback make every NIM run non-negative-by-construction and oracle-free-by-construction.*
