# RESULTS — W112 Lane α: stronger-model reachability sweep ⇒ gate OPEN; Llama-4-Maverick selected; 405B 6th-404

**Verdict: Lane α gate = OPEN. For the FIRST time since W104, a genuinely
STRONGER-than-70B code-capable, same-budget-comparable model is REACHABLE on NIM
— so W112 is NOT defeat-by-default and NOT a 70B rerun. Per the LOCKED § 1α
ranking rule, the primary target is `meta/llama-4-maverick-17b-128e-instruct`
(Llama-family cross-generation-UP). 405B returned its SIXTH consecutive 404.**

This note records the honest reachability/availability pass + target selection
(`docs/RUNBOOK_W112.md` § 1α, locked BEFORE the probe). The earned pilot's
empirical verdict is in `docs/RESULTS_W112_STRONGER_MODEL_BIGCODEBENCH_PILOT_V1.md`.

---

## 1. The sweep (sub-second probes; effectively $0)

`scripts/run_w112_stronger_model_reachability_sweep_v1.py` GET the live NIM
`/v1/models` catalogue (**118 models served**), then probed 13 candidates with a
`max_tokens=4` chat-completion (the W105 probe body). It does NOT assume 405B
exists; it enumerates what NIM actually serves at run time.

**Decision CID `a654956bf0577d78…`;
`results/w112/stronger_model_reachability/sweep_*/sweep_decision.json`.**

| Target | Probe | Eligible? (C-S1∧C-S2∧C-S3∧C-S4) | Note |
|---|---|---|---|
| `meta/llama-3.1-405b-instruct` | **HTTP 404** | no (unreachable) | **6th consecutive 404** (W104–W108, W112); standing extension still dead |
| **`meta/llama-4-maverick-17b-128e-instruct`** | **HTTP 200** | **YES (tier 1)** | **SELECTED** — Llama-family cross-generation-UP; frontier MoE 400B total/17B active; non-reasoning |
| `qwen/qwen3-coder-480b-a35b-instruct` | HTTP 200 | YES (tier 2) | strongest reachable CODE-specialized frontier; cross-vendor ⇒ W113 escalation/cross-check |
| `deepseek-ai/deepseek-v4-pro` | HTTP 200 | YES (tier 2) | DeepSeek V4-pro (V-series chat, non-reasoning) |
| `mistralai/mistral-small-4-119b-2603` | HTTP 200 | YES (tier 2) | 119B non-reasoning instruct |
| `mistralai/mistral-large-3-675b-instruct-2512` | timeout (20 s) | no (unreachable at run time) | frontier 675B; did not respond |
| `qwen/qwen3.5-397b-a17b` | timeout (20 s) | no | 397B; did not respond; also reasoning-mode C-S3 risk |
| `nvidia/nemotron-4-340b-instruct` | HTTP 404 | no | unreachable |
| `mistralai/mistral-large-2-instruct` | HTTP 404 | no | unreachable |
| `nvidia/llama-3.1-nemotron-ultra-253b-v1` | HTTP 404 | no | reasoning-by-default C-S3 EXCLUDE + unreachable |
| `deepseek-ai/deepseek-r1` | HTTP 404 | no | reasoning C-S3 EXCLUDE + unreachable |
| `nvidia/llama-3.1-nemotron-70b-instruct` | HTTP 404 | no | NOT strictly stronger (70B) |
| `meta/codellama-70b` | HTTP 404 | no | NOT strictly stronger (70B) |

**4 eligible reachable stronger targets** (Maverick + Qwen3-Coder-480B +
DeepSeek-V4-pro + Mistral-Small-4-119B).

---

## 2. Target selection — the LOCKED § 1α rule, applied honestly

The § 1α rule (locked before probing) ranks: **tier 1 = Llama-FAMILY larger
instruct, non-reasoning** (cleanest cross-generation-UP from the Llama-3.3-70B
baseline; consistent with the W104 cross-generation precedent) **> tier 2 =
cross-architecture strictly-larger non-reasoning instruct.**

* The only tier-1 eligible target is **`meta/llama-4-maverick-17b-128e-instruct`**
  — the Llama-family successor (Meta Llama 3.3 → 4), frontier MoE, non-reasoning,
  genuinely stronger than Llama-3.3-70B on code. (The cleaner same-family
  *only-scale-changes* probe, 405B, is 404 for the 6th time.) **SELECTED.**
* The strongest ABSOLUTE reachable code model is `qwen/qwen3-coder-480b-a35b-
  instruct` (code-specialized 480B). Under the locked rule it is tier 2
  (cross-vendor confound: a PASS could be "Qwen code-training" not "scale"). It
  is **pre-committed as the W113 escalation / cross-check** — not ignored, but not
  the primary, to honour the pre-locked ranking and isolate the cleanest scale
  signal first.

This is the disciplined outcome: the rule was locked BEFORE the catalogue was
seen, precisely to prevent post-hoc target-shopping. Maverick satisfies the
tier-1 "Llama-family larger instruct" preference; the stronger-absolute
cross-vendor models are the documented next step.

### Same-budget comparability (C-S3) is genuine, not assumed

Maverick is `-instruct` (not `-thinking`/`-reasoning`): it emits plain
completions, so the K=5 self-consistency + reflexion budget is byte-exact
comparable to the 70B bench. The §1α-earn CANARY (2 problems, 22 calls, wall
187 s) CONFIRMED this empirically: parseable ```python``` solutions, executor
ran, 100 % on the two attempt-0 controls, no `max_tokens` overflow / reasoning
trace. MoE active-param count (17B) does not break the call/token budget
invariant the comparison rests on. Reasoning models (DeepSeek-R1, Nemotron-Ultra,
Qwen3.5-thinking) were excluded under C-S3 (budget non-comparable) — and were
unreachable anyway.

---

## 3. Carry-forward

**Added:**
* `W112-T-405B-GATE-SIXTH-404-CLOSED` — `meta/llama-3.1-405b-instruct` re-probed
  at W112 (after locking the runbook) → HTTP 404, the SIXTH consecutive
  (W104–W108, W112). Refreshes `W104-L-…-405B-UNREACHABLE-ON-NIM-CAP`.
* `W112-T-STRONGER-CODE-MODEL-REACHABLE-LLAMA4-MAVERICK` — for the first time
  since the cross-scale-UP axis opened, a strictly-stronger, same-budget-
  comparable, non-reasoning code-capable model is reachable on NIM
  (`meta/llama-4-maverick-17b-128e-instruct`, HTTP 200); the cross-scale-UP
  resistant-code gate is LIVE (decision CID `a654956b…`). 3 further eligible
  tier-2 targets (Qwen3-Coder-480B, DeepSeek-V4-pro, Mistral-Small-4-119B) are
  reachable; Qwen3-Coder-480B is the pre-committed W113 escalation/cross-check.

**Not retired:** the two confirmed retirements (W89, W105) — unchanged. The gate
being OPEN does NOT itself change any claim; the earned pilot's verdict does.
