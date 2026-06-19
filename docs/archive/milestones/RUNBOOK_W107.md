# W107 — 405B reachability gate + next-code-battlefield β preflight + publication-grade consolidation (runbook)

> **Pre-commit contract for W107, locked 2026-05-28 BEFORE any
> W107 NIM call — including the cheap sub-second 405B reachability
> probe that decides the α/β gate.  This runbook is locked FIRST so
> that whatever the probe returns, the next step is pre-committed,
> not decided post-hoc.**
>
> W106 (`COO-30`; Done) closed with: the bounded SECOND retirement
> REGISTERED (`W106-T-BOUNDED-SECOND-RETIREMENT-REGISTERED`); the
> Llama-3.1-70B margin-cap branch CLOSED **NO-GO**
> (`W106-L-HUMANEVAL-PLUS-LLAMA31-70B-MARGIN-CAP-CHEAP-CONFIRMATION-NOT-EARNED-CAP`;
> $0 NIM); `coordpy.margin_cap_dispatch_v1` shipped; graphify
> refreshed at start + close.  405B re-probed at W106 close =
> **HTTP 404**.  `COO-9` REMAINS the lead path.
>
> W107 is prefigured verbatim by `docs/RUNBOOK_W106.md` § 7.  It is
> a **gated branch milestone with THREE lanes**, NOT a new broad
> benchmark tournament:
>
> * **Lane α — 405B reachability gate.**  Re-probe
>   `meta/llama-3.1-405b-instruct` on NIM.  Reachable ⇒ the cheapest
>   honest HumanEval+ cross-scale-UP Phase 2 pilot.  HTTP 404 ⇒
>   record the gate sharply and branch to β as the main empirical
>   lane.
> * **Lane β — next-code-battlefield NIM-free preflight.**  The
>   default empirical lane if 405B stays blocked.  LiveCodeBench
>   primary / APPS backup (the W106 pre-commit).  Serious NIM-free
>   preflight + selection path; NO new vibes tournament; NO new NIM
>   spend until a preflight + cheap pilot earns it.
> * **Lane γ — publication-grade W89→W106 consolidation.**  Runs in
>   parallel; spends $0 NIM; sharpens the claim surface so the
>   boundedness of W105 is impossible to miss.
>
> No version bump.  No PyPI publish.  `coordpy.__version__` stays
> `0.5.20`; `coordpy.SDK_VERSION` stays `coordpy.sdk.v3.43`;
> `coordpy/__init__.py` untouched.  Any new W107 module is
> explicit-import only.

## Linear

* New issue **`COO-31`** (W107): 405B gate + next-code-battlefield
  β preflight + publication-grade consolidation.  Parent: `COO-6`.
  High priority.
* Related: `COO-9` (lead path) — remains at High.  W107-β is the
  direct execution of the `COO-9` charter DoD on the
  post-EvalPlus battlefield (pick a family + justify, build a
  loader/evaluator with the same fairness discipline, specify
  A0/A1/B before running, produce a runbook for the first attack).
* Related: `COO-30` (W106; Done) — the bounded-retirement
  registration W107 builds on.
* Related: `COO-14` (Done) — `coordpy.code_slice_selector_v1`,
  the slice-selection helper integrated into the β preflight.
* Parent `COO-6` backlog: append a W107 snapshot comment at close
  (per `docs/LINEAR_GITHUB_SYNC.md`; the parent description stays
  an append-only comment thread, not rewritten in place).

## What is NOT in scope (anti-drift contract, carried verbatim from W105/W106 + sharpened)

W107 explicitly does NOT:

1. Run a new benchmark-family tournament.  The W101 battlefield-
   selection matrix is the locked rubric; W107-β applies it to the
   two W101-RESERVED candidates (LiveCodeBench / APPS), it does not
   re-open the EvalPlus pair.
2. Re-open MBPP+ V2.  `W102-L-MBPP-PLUS-V2-REFLEXION-PHASE2-70B-CAP`
   stands.
3. Re-open the frozen cross-modal lines.  RealWorldQA stays frozen
   at 11B (W100); MathVista margin-capped; ChartQA preflight-
   saturated.
4. **Re-run any rescue-concentrated Llama-3.1 cheap confirmation.**
   W106 CLOSED that branch NO-GO
   (`W106-L-...-CHEAP-CONFIRMATION-NOT-EARNED-CAP`).  It must NOT be
   re-introduced under a different label.  The ONLY live way to
   strengthen the Llama-3.1 result is a genuinely different
   battlefield (cross-scale-UP), never a rescue-concentrated re-run.
5. Run a 405B EXPENSIVE bench.  A cheap sub-second 405B
   reachability probe is allowed (it decides the α/β gate).  A 405B
   cheap PILOT is allowed ONLY if the gate returns reachable AND the
   § 3 cheap-pilot earning rule is satisfied; a 405B Phase 3 bench
   is never launched in W107 (it is W108-α at the earliest).
6. Bump `coordpy.__version__` or `SDK_VERSION`.
7. Publish to PyPI.
8. Edit `coordpy/__init__.py`.  Any new W107 module is
   explicit-import only.
9. Re-introduce any anti-pattern under a prettier name (bounded
   windowing; compaction; generic prose summarization; shallow
   token compression; context-pruning theater; "cram less /
   truncate better").  The W97–W106 frontier-relevance audits stay
   in force verbatim.  The β preflight selects a HARDER battlefield;
   it does not select a smaller window.
10. Over-state the consolidation.  Lane γ registers exactly what
    W89→W106 already earned: TWO confirmed retirements, both at 70B
    on `meta/llama-3.3-70b-instruct`; NOT cross-class; NOT
    cross-scale-UP; NOT MBPP-family; NOT cross-modal; NOT "context
    solved".

## Operational state (cheap evidence in hand BEFORE W107 starts)

| Field | Value |
|---|---|
| `coordpy.__version__` | `0.5.20` |
| `coordpy.SDK_VERSION` | `coordpy.sdk.v3.43` |
| Confirmed retirements | TWO: W89 (base HumanEval, +5.56 pp) + W105 (HumanEval+, +7.00 pp), both `meta/llama-3.3-70b-instruct` @ 70B |
| W105 Llama-3.3-70B Phase 3 | **RETIRED** (6/6 bars; +7.00 pp; MLB-2 = 55.62 %) |
| W105 Llama-3.1-70B Phase 3 | **FAIL_MARGIN** (5/6; +2.33 pp; MLB-2 = 50.54 %); branch CLOSED NO-GO at W106 |
| W105 Phase 3 slice pack CID | `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314` |
| W105 inner kernel | 30 problems at the head of the helper-priority order (byte-equal to the W103/W104 cheap-pilot slice) |
| HumanEval+ corpus SHA-256 (LFS oid) | `908377f1daf28dcb36846db73a5662b2e05a9907407c2696c89ad9d3b0b04492` |
| 405B reachability (W104 / W105 / W106) | HTTP 404 — unreachable on NIM, three consecutive probes |
| graphify graph build commit (W107 start) | `a560202` (73 199 nodes / 237 569 edges / 2 364 communities) |
| `NVIDIA_API_KEY` present at W107 lock | YES — so the 405B gate probe WILL be re-run live this milestone |

## 1. α / β / γ branch logic (LOCKED)

```
LOCK this runbook (done, BEFORE any NIM call)
        |
        v
[ Lane α gate ] re-run scripts/run_w105_405b_reachability_probe.py
        |
        +-- HTTP 200 (reachable) --> Lane α LIVE:
        |        evaluate the § 3 cheap-pilot earning rule.
        |        EARNED  -> run the HumanEval+ cross-scale-UP Phase 2
        |                   cheap pilot at meta/llama-3.1-405b-instruct
        |                   on the W105 inner-kernel slice (§ 3).
        |        NOT EARNED (gate-rule fail) -> record why; do NOT run;
        |                   fall through to Lane β.
        |
        +-- HTTP 404 / other (unreachable) --> Lane α CLOSED for W107:
                 record the gate result sharply (carry-forward refresh);
                 Lane β becomes the main empirical lane.

[ Lane β ] (default if α not live) build the NIM-free preflight +
        selection path for the next code battlefield (§ 4).
        NO NIM spend in β.

[ Lane γ ] (always, parallel, $0 NIM) publication-grade
        W89->W106 consolidation (§ 6).
```

Lanes β and γ ship UNCONDITIONALLY (they cost no NIM).  Lane α's
pilot is the ONLY part gated on a live NIM call, and it is doubly
gated (reachable AND earning-rule PASS).

## 2. 405B reachability gate rule (LOCKED)

* **Probe**: `python scripts/run_w105_405b_reachability_probe.py`
  (reused byte-for-byte; sub-second; writes
  `results/w107/405b_reachability_probe/probe_<ts>/probe.json`
  via `--out-root`).  Target = `meta/llama-3.1-405b-instruct`.
* **Gate decision**:
  * `status == "reachable"` (HTTP 200) ⇒ **GATE = OPEN** ⇒ Lane α
    becomes LIVE; proceed to the § 3 earning rule.
  * `status == "http_error"` with `http_status == 404` ⇒ **GATE =
    CLOSED** ⇒ refresh
    `W104-L-HUMANEVAL-PLUS-CROSS-SCALE-UP-PRIMARY-TARGET-405B-UNREACHABLE-ON-NIM-CAP`
    with the W107 probe timestamp; Lane β is the main lane.
  * any other status (`http_error` non-404, `exception`,
    `no_api_key`) ⇒ **GATE = CLOSED (indeterminate)** ⇒ treated
    exactly like 404 for branch purposes (no α pilot); the raw
    status is recorded verbatim for the public record.
* The gate probe does NOT change any W107 matrix by itself; it only
  selects which lane is live.  No matrix is widened to include 405B
  unless the gate is OPEN and the § 3 rule is satisfied.

## 3. Cheap-pilot rule if Lane α becomes live (LOCKED — mirrors W104 byte-for-byte)

If GATE = OPEN, the W107-α cheap pilot is the cross-scale-UP analogue
of the W104 cross-generation pilot:

* **Slice**: the W105 Phase 3 slice pack inner kernel = the 30
  problems at the head of the helper-priority order, reused
  BYTE-FOR-BYTE (same slice the W103/W104 cheap pilots used).  Pack
  CID `8be55f3bf1650df3…`.  NO new slice is constructed.  This is
  the byte-equal W103 helper-anchored slice the W106 § 7 contract
  names; no stronger pre-committed reason exists to deviate, so it
  is used unchanged.
* **Arms (A0 / A1 / B), byte-identical to W103/W104**:
  * **A0** — single-shot zero-temperature baseline (1 sample;
    `pass@1` on the raw prompt; no reflexion).
  * **A1** — same-budget best-of-K single-agent baseline at K=5
    (`pass@1` over 5 independent samples; the strong single-agent
    arm the retirement must beat).
  * **B** — W89 sequential-reflexion team at the SAME K=5 budget
    (read → solve → execute → reflect → repair), byte-exact budget
    parity with A1.
* **Target**: `meta/llama-3.1-405b-instruct` (cross-scale-UP from
  the 70B retirement; the genuine strengthening axis).
* **Budget**: 1 seed × 30 problems × K=5 × 2 arms (+ A0) ≈ 330 NIM
  calls at the 405B rate.  This is the W93/W104 cheap-pilot
  envelope.
* **Gates (the W104 Phase 2 9-gate + MLB sub-gate shape, verbatim
  from `scripts/run_w104_humaneval_plus_cross_scale_pilot.py`
  `_evaluate_phase2_gates`)**:
  * G1 slice pre-committed; G2 A1 < 90 %; G3 B > A1; G4 (B − A1) ≥
    +5 pp; G5 (B − A0) ≥ +5 pp; G6 per-problem majority (B ≥ A1 on
    ≥ 16 of 30); G7 budget byte-exact; G8 audit chain re-derives;
    G9 executor clean.
  * MLB-1 invocation rate ≥ 33 %; MLB-2 rescue rate ≥ 33 %.
  * **Verdict** = `PASS_MECHANISM_DRIVEN` iff 9/9 gates PASS AND
    both MLB sub-gates PASS; `PASS_NON_MECHANISM_DRIVEN` iff 9/9 but
    an MLB sub-gate fails; else `FAIL`.
* **Cheap-pilot earning rule (LOCKED)**: the α pilot is RUN iff GATE
  = OPEN.  It EARNS the W108-α Phase 3 cross-scale-UP retirement
  bench iff its verdict is `PASS_MECHANISM_DRIVEN`.  Any other
  verdict ⇒ record a `W107-L-HUMANEVAL-PLUS-CROSS-SCALE-UP-405B-<reason>-CAP`
  carry-forward and do NOT widen; cross-scale-UP stays unconfirmed.
* **Cross-scale-collapse honesty**: a single-scale +X pp at 405B is
  an UPPER BOUND on the broad-slice Phase 3 margin (the W102/W104/
  W96-A/W96-C/W100 lesson).  The α pilot can EARN a Phase 3 bench;
  it can NEVER by itself constitute a cross-scale-UP retirement.

## 4. LiveCodeBench / APPS selection rule (LOCKED — Lane β)

The W106 § 7 pre-commit names **LiveCodeBench primary / APPS
backup**.  W107-β does NOT re-litigate that ordering by vibes; it
applies the W101 C1–C8 rubric (`docs/RESULTS_W101_BATTLEFIELD_SELECTION_V1.md`)
to the two W101-RESERVED candidates and pre-commits a STRUCTURAL-
SOUNDNESS test that can pivot primary→backup inside this milestone.

### 4.1 The pre-committed order and why

At W101 both LiveCodeBench and APPS were ranked "out of scope"
**relative to the EvalPlus pair** (MBPP+/HumanEval+), not
disqualified (only SWE-bench-lite took an F).  Now that the
EvalPlus pair is exhausted — MBPP+ capped (W102), HumanEval+
retired single-class (W105) — the next battlefield is exactly the
W101-RESERVED pair.  LiveCodeBench leads because its **time-anchored
contamination resistance** (C7) is the single most decisive property
for a *publication-grade* multi-agent-superiority claim: it is the
only candidate whose design directly answers "is the retirement
real or training-set contamination?".  APPS is the cleaner-executor,
heavier-corpus backup.

### 4.2 Structural-soundness test (LOCKED — decides primary vs backup THIS milestone)

LiveCodeBench REMAINS primary unless the NIM-free preflight shows it
is **structurally wrong** on any of these three hard gates, in which
case W107-β PIVOTS to APPS in the same milestone (no extra
milestone of paperwork):

* **S1 — executor cleanness reachable (C2).**  A deterministic,
  no-LLM-judge, binary PASS/FAIL executor must be constructible for
  a coherent subset.  Both candidates carry stdin/stdout *and*
  functional/call-based problems; the W89 mechanism produces a
  complete function, so the clean subset is the functional/call-based
  form.  S1 FAILs only if no clean subset of adequate size exists.
* **S2 — NIM-free failure-residual estimate exists (C1, C6).**  A
  Phase-2 A1@K=5 residual estimate must be computable WITHOUT a new
  NIM call.  Unlike the EvalPlus pair (which re-executed existing
  W88/W91 sidecars against new tests), neither reserved candidate
  has a local sidecar of its own problems, so the residual estimate
  is **published-baseline-grade** (leaderboard pass@1 for
  Llama-3.x-70B), explicitly weaker than re-executed-sidecar-grade.
  S2 PASSES iff a published or operator-verifiable A1@K=5 estimate
  ≤ 90 % (or a documented residual ≥ +10 pp) exists for the chosen
  subset.  S2 FAILs only if no honest NIM-free residual can be
  stated at all.
* **S3 — W89 decomposition fit (C3).**  The W89 read→solve→
  execute→reflect→repair shape must port to the chosen subset
  without becoming a different research project.  Functional/
  call-based problems fit (same "produce a complete function"
  shape); pure interactive/stdout-streaming problems do not.  S3
  FAILs only if the clean subset cannot be attacked by the existing
  W89 mechanism.

**Decision**: primary = LiveCodeBench iff S1 ∧ S2 ∧ S3 hold for a
LiveCodeBench functional subset; else primary = APPS iff S1 ∧ S2 ∧
S3 hold for an APPS subset; else the post-EvalPlus code battlefield
is structurally capped and W108 records an honest no-go (§ 7).

### 4.3 β deliverables (LOCKED — what W107-β ships, all NIM-free)

The β lane ships a serious preflight + selection path, enough that
W108 is a cheap pilot or an honest no-go, not paperwork:

1. **Exact battlefield-selection rule** — § 4.1 + § 4.2 above,
   applied with a written C1–C8 + S1–S3 verdict per candidate.
2. **Loader scaffolding for the lead** — a SHA-pinnable corpus
   loader that mirrors `coordpy.humaneval_plus_loader_v1` (canonical
   URL pin; refuses unpinned/mismatched corpus; read-only; no NIM),
   restricted to the functional/call-based subset.
3. **Evaluator (executor) scaffolding** — a deterministic
   subprocess executor mirroring `coordpy.humaneval_plus_executor_v1`
   cleanness invariants (fresh `-I` CPython subprocess; soft +
   kill wall timeout; binary PASS/FAIL on exit 0; stderr/stdout tail
   returned for reflexion signal; NO LLM judge), handling the
   call-based test-input/expected-output shape.
4. **Exact A0 / A1 / B definitions** — byte-identical to the
   W103/W104/§ 3 arms (A0 single-shot; A1 best-of-K=5; B W89
   sequential reflexion at the same K=5 budget).
5. **Cheap integrity probes** — corpus integrity (SHA pin), executor
   self-test on a gold/canonical solution (must PASS) + on a known-
   wrong solution (must FAIL), subset-size adequacy (≥ 30 problems
   with a clean executor), and a NIM-free A1@K=5 residual estimate.
6. **Executor cleanness check** — the S1 verdict with the concrete
   executor shape.
7. **Failure-residual estimate** — the S2 published-baseline-grade
   A1@K=5 estimate, with operator-verification flagged where it
   relies on an external leaderboard number rather than a re-executed
   local sidecar.
8. **Decomposition argument** — the S3 verdict (W89 shape ports to
   the functional subset).
9. **Helper/slice-selection integration** — wire
   `coordpy.code_slice_selector_v1` (COO-14; Done) to propose the
   cheap-pilot slice on the chosen battlefield, exactly as it did for
   the W103 HumanEval+ slice, IF it materially helps the selection
   (else record why not).

β does NOT launch any pilot.  The cheap pilot is W108 work, gated on
the β preflight verdict + the W93 5-gate discipline.

## 5. graphify deliverables (LOCKED)

* **Initial refresh** (DONE at W107 start): `graphify update .`
  rebuilt the graph from current HEAD `a560202` (73 199 nodes /
  237 569 edges / 2 364 communities); the watcher reported "no
  code-graph topology changes" because the working tree equals HEAD.
* **Concrete usage during the milestone**: `graphify query`
  (HumanEval+ bounded-retirement claim provenance + doc truth
  surfaces); `graphify path` (`humaneval_plus_reflexion_bench_v1.py
  --imports_from--> humaneval_plus_executor_v1.py`, the exact
  loader→executor→bench wiring chain the β scaffolding mirrors);
  `graphify explain code_slice_selector_v1` (the COO-14 helper that
  references HumanEval+/MBPP+ and is implemented-by the W102
  arsenal-mining pilot — the slice helper β integrates);
  `graphify affected` (reverse-dep probe of the loader/executor
  surfaces).  Finding: the code-graph is AST-built (no semantic LLM
  key set here), so doc-layer truth surfaces are navigated directly;
  the new β scaffolding sits as a SIBLING of the
  `humaneval_plus_*` loader/executor/bench triple it mirrors.
* **End-of-milestone refresh** (REQUIRED before close): re-run
  `graphify update .` after the material code/doc edits so
  `graphify-out/` matches W107 repo truth; record the new build
  commit; keep `graphify-out/` as a local operating surface (not
  forced into tracked history).

## 6. Publication-grade consolidation (LOCKED — Lane γ, $0 NIM)

Lane γ is real work, not fluff.  With TWO confirmed retirements and
a bounded claim structure, γ sharpens the truth surface so future
ambiguity is reduced and the claims are easier to defend:

* **Consolidated narrative doc** —
  `docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md`: the W89 →
  W103 → W104 → W105 → W106 code-retirement arc as one defensible
  narrative (this is the W105 Verdict-A hardening-lane content,
  deferred at W106 because the realized verdict was C/bounded).
* **Claim-surface harmonization** — verify the registry / status /
  honesty docs agree, verbatim, on: TWO confirmed retirements (W89
  + W105), both `meta/llama-3.3-70b-instruct` @ 70B; the W105 split;
  the three-axis boundedness (class / family / scale); and the
  five explicit non-claims (cross-class / cross-scale-UP /
  MBPP-family / cross-modal / "context solved").
* **Boundedness front-and-centre** — the consolidation makes the
  single-class boundedness impossible to miss; it must NOT read as a
  louder claim than W105/W106 already licensed.

## 7. W108 branch logic (LOCKED — make the next milestone execution, not paperwork)

W108 is decided by the realized W107 outcome:

* **If Lane α ran and PASSed mechanism-driven** — W108-α =
  HumanEval+ Phase 3 cross-scale-UP retirement bench at
  `meta/llama-3.1-405b-instruct` (3 seeds × 100 problems × K=5 on
  the W105 pack; the genuine single-class → cross-scale
  strengthening).  This would be the path to the programme's first
  cross-scale retirement.
* **If Lane α ran and did NOT pass** — W108 = the β battlefield
  cheap pilot (cross-scale-UP recorded as attempted-but-unconfirmed
  via the § 3 carry-forward).
* **If Lane α was CLOSED (405B unreachable) and the β preflight
  PASSed for the primary (LiveCodeBench, or APPS on a structural
  pivot)** — W108 = the cheap Phase 2 pilot on that battlefield
  (1 seed × 30 problems × K=5; ~330 NIM calls; the W93 5-gate
  discipline + § 3-shape gates apply).
* **If the β preflight FAILed S1∧S2∧S3 for BOTH LiveCodeBench AND
  APPS** — W108 = honest no-go: the post-EvalPlus code battlefield
  is structurally capped; the next live move is one of {`COO-12`
  substrate-level cross-modal injection promotion, a 405B re-probe
  if infrastructure changes}, decided explicitly, NOT a re-run of
  any capped/frozen line.

`COO-9` stays the lead path under every W108 branch unless the β
preflight forces a documented code-line move; the bounded second
retirement (W105/W106) is its strongest registered claim.

## 8. Stable boundary preservation

* `coordpy.__version__` unchanged at `0.5.20`.
* `coordpy.SDK_VERSION` unchanged at `coordpy.sdk.v3.43`.
* No PyPI publish.
* `coordpy/__init__.py` untouched.
* Any new W107 module is explicit-import only.
* New W107 artefacts (this milestone):
  * `docs/RUNBOOK_W107.md` (this file).
  * `scripts/run_w107_405b_reachability_gate.py` (thin wrapper that
    runs the probe under the W107 out-root + records the gate
    decision; reuses the W105 probe logic).
  * Lane β scaffolding (NIM-free; explicit-import-only) — the lead
    battlefield loader/executor preflight modules + a preflight
    driver + the β preflight RESULTS doc.
  * `docs/CONSOLIDATED_CODE_RETIREMENT_NARRATIVE_V1.md` (Lane γ).
  * `docs/RESULTS_W107_*` (gate verdict, β preflight, milestone
    summary) + `docs/FRONTIER_RELEVANCE_AUDIT_W107_V1.md` (17th
    preflight-discipline validation).

## Honest framing

W107's job is to:

1. **Decide the 405B gate honestly and cheaply** — re-probe, record
   the result sharply, and branch; do not pretend α is live if it is
   404 for the fourth time.
2. **Make the next code battlefield real, not vibes** — a serious
   NIM-free preflight + selection path on LiveCodeBench (primary) /
   APPS (backup) with a pre-committed structural-soundness pivot, so
   W108 is a cheap pilot or an honest no-go.
3. **Consolidate the two confirmed retirements at publication grade**
   — sharpen the claim surface, keep the boundedness impossible to
   miss, spend $0 NIM.
4. **Keep graphify in the loop** — refresh, query, refresh.

The programme is entitled to exactly the claim W105/W106 licensed —
TWO confirmed retirements, both at 70B on
`meta/llama-3.3-70b-instruct`, bounded on three axes — and W107 does
nothing to inflate it.  A cross-scale-UP retirement is the genuine
strengthening path and is gated on 405B becoming reachable; until
then the β preflight builds the only other honest path to a stronger
claim: a third code-benchmark family attacked under preflight-first
discipline.
