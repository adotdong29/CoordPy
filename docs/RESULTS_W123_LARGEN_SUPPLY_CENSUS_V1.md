# RESULTS — W123 Lane-α: official ICPC large-n supply census

**Date:** 2026-05-31 · **Verdict:** `LARGEN_MATCHED_BATTLEFIELD_UNREACHABLE_OFFICIAL_FAMILY` · **census_cid** `4d6f4e8f1a2c`
**Spend:** $0 NIM (Lane β gate did not open). Decision CID `258b6ed7` invariant. No version bump, no PyPI, `coordpy/__init__.py` untouched.

## What W123 attempted

W122 left the matched resistant-vs-exposed ICPC contrast **unresolvable at n=30** (3-seed means resistant +4.44pp / exposed +8.89pp, B4). The pre-committed escalation was **larger n PER FIELD (≥100/field)** on the SAME official `github.com/icpc` family. W123 Lane α asks, deterministically and NIM-free: *can that family supply ≥100 tier-1 pure pass-fail tasks on BOTH sides?*

## Method (NIM-free, evidence-backed)

A pinned census of every official-org problem-package surface, **re-verified live via the GitHub API on 2026-05-31** (`verified_live=True, all_match=True`): RMRC repos counted by `problem.yaml`; the `na-ecna-archive` counted by per-year `*.zip` Kattis packages. Module `coordpy.icpc_largen_supply_census_v1`; script `scripts/run_w123_largen_supply_census_v1.py`.

## The supply census

| side (vs cutoff 2024-08-31) | package-bearing surfaces | raw problems | est. tier-1 | ≥100? |
|---|---|---|---|---|
| **RESISTANT** (post-cutoff) | RMRC 2024-25 (13), RMRC 2025-26 (13), ECNA 2024-25 (13), ECNA 2025-26 (12) | **51** | ~45 (W120 actual 45) | **NO** |
| **EXPOSED** (pre-cutoff) | RMRC 2017/18/19/20/21/22-23 (11+10+11+11+14+12=69) + ECNA 2019-20…2023-24 (17+12+13+12+12=66) | **135** | ~113 | **YES** |

**Excluded org repos (ship no Kattis packages):** `na-rocky-mountain-2023-2024-public` (**0** `problem.yaml`) and `na-mid-atlantic-public` (README-only stub). (`*-web` repos are websites.)

Consistency anchor: the resistant raw total **51 == W120's recorded `n_seen` of 51** — the census exactly reproduces W120's surface footprint.

## Finding

- **The resistant side is hard-capped at 51 raw / ~45 tier-1 — below 100 even at a 100% yield.** Exactly **4** post-cutoff official package surfaces exist in the org, and **W120 already mined all four**. There is **no fifth** post-cutoff `github.com/icpc` package surface.
- **The exposed side scales past 100** (135 raw / ~113 tier-1 across 11 pre-cutoff surfaces; W121 mined only 4 of them, so there is large unmined headroom — RMRC 2017/18/19/20 + ECNA 2019-20/2020-21/2021-22).
- Therefore the **≥100/field MATCHED battlefield cannot be built**, and the blocker is **solely the post-cutoff (resistant) axis**: real-world contest-package supply (only four post-cutoff regional seasons have been published as graded packages), not the rule, the grader, or curation.

## Consequence for the contamination question

W122's "unresolvable at n=30" is a genuine **power** problem, but the SAME official family **cannot supply the resistant n** needed to resolve it (the exposed n is available; the resistant n is not). The matched-family caveat therefore **stands, with its blocker now machine-checkable and precisely localized**: it is **post-cutoff-supply-bound**, not method-bound. This is registered as a limitation; it does **not** retire and does **not** strengthen any claim.

## What did NOT happen (and why)

- **No Lane β large-n pilot, $0 NIM.** Gate requires BOTH fields ≥100 from the official family; the resistant side cannot reach it. (`largen_spend_gate_open=False`.)
- **No second n=30 seed** (anti-pattern; W122 capped at three).
- **No new battlefield family, no dirty exposed benchmark, no 405B, no mechanism/M3 reopening.**

## Carry-forward (unchanged)

Exactly **TWO** confirmed retirements stand — **W89** (base HumanEval ×llama-3.3-70b, +5.56pp) and **W105** (HumanEval+ ×llama-3.3-70b, +7.00pp), both contamination-EXPOSED HumanEval-family at 70B. Resistant superiority remains **0 clean** across W108/W110/W113/W120. Stronger-model gate **STRUCTURALLY CLOSED + MOOT** `{KNOWN:1, UNKNOWN:4}` (Lane γ).
