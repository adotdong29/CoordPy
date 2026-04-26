# RESULTS — Wevra capsule-native runtime (SDK v3.1)

> *Theory-forward results note. States the claim, the code anchors,
> the empirical status, the proof boundary, and what remains
> conjectural. Last touched: 2026-04-26.*

---

## 0. One-paragraph summary

Up to SDK v3 the capsule layer was a **post-hoc audit fold**: a
Wevra run executed end-to-end as ordinary Python, the
``product_report`` dict was handed to ``build_report_ledger`` at the
end, and a sealed capsule DAG was synthesised after the fact.
Capsules described the run; they did not gate it. SDK v3.1 takes
the next step: the runtime itself drives execution through capsule
lifecycle transitions. Each runtime stage — profile resolution,
readiness check, sweep-spec admission, per-cell execution,
provenance manifest, substantive artifact emission, and the final
run-report — corresponds to a capsule transition that happens *in
flight*. Three theorem-style claims follow: lifecycle ↔ execution-
state correspondence (W3-32, proved by inspection of
``CapsuleNativeRunContext`` + contract test), content-addressing
at artifact creation time (W3-33, proved by inspection of
``seal_and_write_artifact`` + cross-validation test), and
ledger-equivalence between in-flight and post-hoc paths on the
non-ARTIFACT kinds (W3-34, proved by per-CID set-equality on a
deterministic smoke run). The two paths produce CID-equivalent
ledgers for PROFILE / READINESS_CHECK / SWEEP_SPEC / SWEEP_CELL /
PROVENANCE / RUN_REPORT; they differ only on the ARTIFACT kind,
where the in-flight path carries real SHA-256 hashes and the
post-hoc fold leaves them ``None`` — an intentional and documented
divergence. The capsule layer is no longer a description of Wevra;
it is now (partially) the **execution contract** Wevra runs under.

---

## 1. The claim in one paragraph

> **Capsules drive the Wevra runtime, not just describe it.** Every
> stage that crosses a run-boundary in a capsule-native run seals
> a typed, content-addressed, lifecycle-bounded, budget-bounded,
> provenance-stamped capsule before the next stage can read its
> result. A stage that fails leaves a typed in-flight register
> entry that never reaches the ledger; downstream stages refuse to
> seal because the parent CID is missing (Capsule Contract C5).
> Substantive on-disk artifacts are content-addressed at write
> time: their SHA-256 is computed before bytes hit disk, baked into
> the sealed ARTIFACT capsule's payload, then verified by re-read.
> The post-hoc ``build_report_ledger`` fold is retained as a
> third-party-facing adapter — the two paths are CID-equivalent on
> non-ARTIFACT kinds (Theorem W3-34) and the legacy SDK-v3 surface
> is unchanged.

---

## 2. Capsule-native execution — the new contract

### 2.1 Lifecycle correspondence (W3-32)

Let $S$ be the runtime's execution stages for one Wevra run:
$\mathcal{S} = \{\mathrm{profile}, \mathrm{readiness}, \mathrm{sweep\_spec},
\mathrm{sweep\_cell}_1, \dots, \mathrm{sweep\_cell}_n, \mathrm{provenance},
\mathrm{artifact}_1, \dots, \mathrm{artifact}_k, \mathrm{run\_report}\}$.

Let $\mathcal{C}_\mathcal{R}$ be the in-flight register of
``CapsuleNativeRunContext`` (a list of ``_InFlightEntry`` records).
Let $\mathcal{L}$ be the runtime's ``CapsuleLedger``.

**Theorem W3-32 (lifecycle ↔ execution-state correspondence).**
For every stage $S_i \in \mathcal{S}$ there is a unique capsule
$c_i \in \mathcal{C}_\mathcal{R}$ with $\mathrm{kind}(c_i) =
\mathrm{kind}(S_i)$ such that:

1. $S_i$ is *in progress* at time $t$ iff $c_i$ has been proposed
   but not yet admitted+sealed at $t$:
   $c_i \in \mathcal{C}_\mathcal{R}$ and
   $\mathit{cid}(c_i) \notin I(\mathcal{L})$.
2. $S_i$ has *completed* at $t$ iff $c_i$ is sealed in
   $\mathcal{L}$ at $t$:
   $\mathit{cid}(c_i) \in I(\mathcal{L})$ and
   $\mathrm{lifecycle}(c_i) = \mathtt{SEALED}$.
3. $S_i$ has *failed* at $t$ iff $c_i.\mathrm{failure} \neq \bot$
   in $\mathcal{C}_\mathcal{R}$ and $\mathit{cid}(c_i) \notin
   I(\mathcal{L})$.

**Proof.** Each stage's seal step is one of the eight methods on
``CapsuleNativeRunContext`` (``start_run`` / ``seal_readiness`` /
``seal_sweep_spec`` / ``seal_sweep_cell`` / ``seal_provenance`` /
``seal_and_write_artifact`` / ``seal_artifact_from_path`` /
``seal_run_report``). Each method follows the same pattern:
``_propose`` (push an ``_InFlightEntry`` with ``failure=None``,
``sealed_cid=None``); call ``_admit_and_seal`` (which calls
``ledger.admit_and_seal``). On admission failure, ``_admit_and_seal``
catches the exception, sets ``entry.failure``, and re-raises;
``sealed_cid`` is never set. On admission success, ``sealed_cid``
is set to the ledger's sealed copy's CID. The ledger's
``admit_and_seal`` is total exactly on the admissibility predicate
$\mathcal{A}_\mathcal{L}$ (Capsule Formalism § 2.3). Therefore the
three conditions above are mutually exclusive and exhaustive on the
runtime's transitions. $\square$

**Code anchor.** ``vision_mvp/wevra/capsule_runtime.py``:
``CapsuleNativeRunContext._propose``, ``._admit_and_seal``, the
eight ``seal_*`` methods. **Empirical anchor.**
``test_w3_32_lifecycle_correspondence_clean_run`` (every kind
present, no failures), ``test_w3_35_cell_refuses_without_spec`` and
``test_w3_35_readiness_refuses_without_profile`` (rejection
witnessed), ``test_failed_admission_leaves_in_flight_entry``
(failure leaves typed register entry, not in ledger).

### 2.2 Content addressing at artifact creation time (W3-33)

Let $A$ be a substantive on-disk artifact written by
``CapsuleNativeRunContext.seal_and_write_artifact`` at path $p$
with bytes $d$, returning a sealed capsule $c_A \in \mathcal{L}$.

**Theorem W3-33 (artifact authenticity by capsule CID).**
At return time of ``seal_and_write_artifact``,

$$
\mathrm{SHA\text{-}256}(\mathrm{read}(p))
\;=\; c_A.\mathit{payload}[\texttt{"sha256"}].
$$

**Proof.** The method's order of operations:
1. $h := \mathrm{SHA\text{-}256}(d)$ on the in-memory bytes.
2. Build $c_A$ with payload $\{\texttt{path}: p, \texttt{sha256}: h\}$;
   admit + seal in $\mathcal{L}$. If admission fails, the method
   raises before any write; the in-flight register records the
   failure (W3-32 case 3).
3. Write $d$ to $p$.
4. $h' := \mathrm{SHA\text{-}256}(\mathrm{read}(p))$.
5. If $h' \neq h$, raise ``ContentAddressMismatch`` (a subtype of
   ``CapsuleAdmissionError``).

If the method returns normally, step 5's check passed, so
$h' = h = c_A.\mathit{payload}[\texttt{"sha256"}]$, which is the
claim. $\square$

**Failure mode.** ``ContentAddressMismatch`` is raised when an
adversarial / racing / corrupting writer overwrites the file
between step 3 (write) and step 4 (re-read). The check is a
TOCTOU detector for honest writers, not a defence against an
adversary with concurrent write access — the trust boundary is
the same as Wevra's sandbox boundary. The in-flight register
records the rejected ARTIFACT capsule's CID; the ledger does NOT
contain it.

**Code anchor.**
``vision_mvp/wevra/capsule_runtime.py::seal_and_write_artifact``;
the ``CapsuleNativeRunContext.seal_and_write_artifact`` method
calls the same logic with parent-CID resolution.
**Empirical anchor.** ``test_w3_33_seal_then_write_then_verify``
asserts the SHA equality post-call; ``test_w3_33_mismatch_detector``
exhibits a corrupted-write scenario raising
``ContentAddressMismatch``.

**What is NOT proved.** W3-33 is silent on:

- The bytes of $d$ being *honest* — the writer's correctness is
  the trust unit. Wevra's substantive writers (readiness verdict,
  sweep result, provenance manifest) compute bytes from JSON
  serialisation of well-typed dicts; their honesty is inherited
  from Python's ``json.dumps``.
- *Adversarial* concurrent writers — see the failure-mode note
  above.
- *Read-side* drift — once a third party reads the file's bytes
  and the capsule's CID, they can verify locally, but the
  in-process re-hash is at write time.

### 2.3 Ledger equivalence on non-ARTIFACT kinds (W3-34)

Let $r$ be a ``product_report`` dict produced by a capsule-native
run. Let $\mathcal{L}_{\rm in} \subseteq r$ be the ledger embedded
in $r[\texttt{capsules}]$ (the in-flight ledger). Let
$\mathcal{L}_{\rm post} := \mathrm{build\_report\_ledger}(r)$ (the
post-hoc fold of the same report).

**Theorem W3-34 (in-flight / post-hoc CID equivalence on
non-ARTIFACT kinds).** For every $k \in \{\texttt{PROFILE},
\texttt{READINESS\_CHECK}, \texttt{SWEEP\_SPEC},
\texttt{SWEEP\_CELL}, \texttt{PROVENANCE}\}$,

$$
\{\mathit{cid}(c) : c \in \mathcal{L}_{\rm in},\, k(c) = k\}
\;=\;
\{\mathit{cid}(c) : c \in \mathcal{L}_{\rm post},\, k(c) = k\}.
$$

**Proof.** For each kind, the in-flight builder and the post-hoc
fold call the *same* adapter (``capsule_from_profile``,
``capsule_from_readiness``, ``capsule_from_sweep_spec``,
``capsule_from_sweep_cell``, ``capsule_from_provenance``) on the
*same* payload (the in-flight runner stores the payload into
$r$ before calling the adapter; the post-hoc fold reads the same
payload back from $r$). C1 (capsule identity) makes
$\mathit{cid}(c)$ a pure function of $(k, p, b, \pi)$. The parent
sets are *also* equal between paths: in both, PROFILE has no
parent; READINESS_CHECK / SWEEP_SPEC / PROVENANCE point to
profile; SWEEP_CELL points to sweep_spec. The runner deliberately
preserves the SWEEP_SPEC payload's shape across paths
(``mode, sandbox, jsonl, model, endpoint, executed_in_process``)
to make this equivalence strict. $\square$

**Empirical anchor.**
``test_w3_34_smoke_run_kind_cid_match`` runs ``local_smoke``
under ``capsule_native=True``, then refolds its
``product_report`` dict via ``build_report_ledger``, then asserts
set-equality of CIDs per equivalence kind. The test runs in
~0.25 s on the bundled bank.

**What is NOT proved (intentional divergence).** W3-34
*excludes* the ARTIFACT kind. The in-flight path's ARTIFACT
capsules carry real SHA-256 payloads
(``payload["sha256"] = <hex>``); the post-hoc fold's ARTIFACT
capsules carry ``payload["sha256"] = None`` because
``build_report_ledger`` does not read the on-disk file. The two
sets are therefore disjoint — a *useful* signal: a capsule view
whose ARTIFACT payloads carry real hashes was produced under the
in-flight path; one whose ARTIFACT payloads carry nulls was
produced post-hoc. ``test_w3_34_artifact_kind_intentional_divergence``
witnesses this disjointness.

W3-34 is also silent on the RUN_REPORT kind. The RUN_REPORT
capsule's ``parents`` field includes every other sealed capsule's
CID; if the ARTIFACT kind diverges, the RUN_REPORT's parent set
diverges, which propagates to its CID. This is consistent with
W3-1 (capsule identity soundness): a different parent set yields
a different CID. Stating an *RUN_REPORT*-level equivalence claim
would require either (i) excluding ARTIFACT capsules from the
RUN_REPORT's parent set under both paths, or (ii) extending
``build_report_ledger`` to compute on-disk SHAs. The honest
position is: RUN_REPORT CIDs across paths are *generally
different*; the durable run identifier under the in-flight path
is the in-flight RUN_REPORT CID, and that capsule's chain is
independently audit-able via ``wevra-capsule verify``.

### 2.4 Lifecycle gating as execution contract (W3-35)

Let $T_S$ be the precondition that a stage $S$ requires before
it can execute (e.g. SWEEP_CELL requires SWEEP_SPEC sealed;
READINESS_CHECK requires PROFILE sealed).

**Theorem W3-35 (parent-CID gating is the execution contract).**
For every stage $S$ with precondition $T_S$, calling the
corresponding ``ctx.seal_*`` method without satisfying $T_S$
raises ``CapsuleLifecycleError`` (when the precondition is a
required prior stage in the runtime context) or
``CapsuleAdmissionError`` (when the precondition is a parent CID
that must already be in the ledger). The ledger does NOT contain
the requested capsule.

**Proof.** Each ``seal_*`` method's first check is the
precondition (``self._require_profile()`` /
``if self.spec_cap is None``). If unsatisfied, the method raises
*before* calling ``_propose``. ``CapsuleLedger.admit`` checks
parent CIDs against ``_by_cid`` (Capsule Formalism § 2.3, C5);
unknown parents raise ``CapsuleAdmissionError``.

**Empirical anchor.**
``test_w3_35_cell_refuses_without_spec``,
``test_w3_35_readiness_refuses_without_profile``,
``test_failed_admission_leaves_in_flight_entry``.

W3-35 is the formal statement of "the runtime cannot execute a
downstream stage unless its upstream prerequisite capsule has been
sealed." This is the *capsule contract* doing the gating that was
previously implicit in Python sequential ordering. The capsule
layer has stopped being mere description and is now (partially)
the runtime's typed execution contract.

---

## 3. What is NOW true about Wevra

The set of statements that *became* true with SDK v3.1:

- **Wevra is a context-capsule runtime, in execution, not just in
  reporting.** Every substantive boundary-crossing artefact in a
  ``capsule_native=True`` run is sealed at boundary-crossing time,
  not folded in afterwards. The capsule view's ``construction``
  field reads ``in_flight``.
- **Substantive artifacts are authenticated by their capsule's
  CID.** A reader of ``readiness_verdict.json`` /
  ``sweep_result.json`` / ``provenance.json`` and the in-flight
  capsule view can verify locally that the on-disk bytes match
  the sealed CID's SHA. Cross-host trust now flows through the
  capsule view, not through path-level conventions.
- **A failed run leaves a typed witness, not a state-bag.** A
  stage that fails (admission rejection, missing parent,
  over-budget capsule) is observable as a ``failed`` entry in
  the in-flight register; downstream stages refuse to seal.
  ``ctx.in_flight_failures()`` returns a list of dicts naming
  the kind / cid / proposed-time / failure of every stage that
  fell over.
- **The post-hoc fold is retained as the third-party adapter.**
  Callers who have a ``product_report`` dict from somewhere
  else — disk, an HTTP API, another runtime — can still call
  ``build_report_ledger`` to lift it under the same Capsule
  Contract. The two paths are CID-equivalent on non-ARTIFACT
  kinds.
- **The SDK v3.0 public surface is unchanged.** Every v3.0
  symbol still imports and behaves byte-for-byte the same. The
  capsule view schema is unchanged
  (``wevra.capsule_view.v1``); the new ``construction`` field
  is additive. The ``RunSpec.capsule_native`` field defaults to
  ``True`` so existing runs flip onto the new path with no
  caller change required.

The set of statements that are *not yet* true:

- **The Wevra runtime is fully capsule-native at every level.**
  Internal sub-steps (parser invocation, sandbox execution, LLM
  call) still pass plain Python objects across function calls.
  The capsule layer captures *run-boundary* objects (profile,
  readiness, sweep spec, cells, provenance, artifacts, report);
  it does not yet capture *intra-cell* objects (the prompt sent
  to the LLM, the patch parsed back, the per-instance test
  result). The next slice would name those as capsules too —
  ``PROMPT``, ``GENERATED_PATCH``, ``TEST_VERDICT`` kinds.
  Phase-49+ research-side work (decoder, bundle policy) is
  orthogonal to this runtime axis.
- **Adversarial concurrent writers are detected.** The post-write
  re-hash is a TOCTOU detector for honest writers. A multi-writer
  scenario with deliberate corruption is out of scope.
- **The legacy ``vision_mvp.product`` module is reformulated.**
  ``vision_mvp.product.runner.run_profile`` now defaults to
  ``capsule_native=True``; the legacy post-hoc path is reachable
  via ``capsule_native=False`` or
  ``--legacy-post-hoc-capsules``. The module's public surface
  is unchanged.

---

## 4. Honest limitations

### 4.1 What the capsule-native path does NOT promise

- **Re-execution determinism.** Two separate runs of the same
  profile produce different PROVENANCE CIDs because the manifest
  carries a timestamp; different RUN_REPORT CIDs because the
  RUN_REPORT's parent set includes the timestamped PROVENANCE
  capsule. PROFILE / SWEEP_SPEC / SWEEP_CELL CIDs *are* stable
  across runs of the same deterministic profile — see
  ``test_run_report_root_cid_stable_across_runs_of_same_profile``.
  Cross-run PROVENANCE / RUN_REPORT stability is not a v3.1
  claim and would require a separate "deterministic-mode"
  toggle that fixes the timestamp.
- **Meta-artifact authentication.** The container files
  ``product_report.json``, ``capsule_view.json``,
  ``product_summary.txt`` are *post-view* renderings of the
  capsule graph; they are not themselves capsule-tracked.
  Naming them under their own ARTIFACT capsules would require
  the capsule view to include capsules that hash its own bytes
  — a circular dependency. The honest position is that
  meta-artifacts are renderings of the canonical capsule view,
  and the canonical view is the source of truth.
- **Full-system content-addressing.** The substantive artifacts
  are content-addressed; the meta-artifacts are not; intra-cell
  artifacts (LLM prompts, parsed patches, sandbox stdout) are
  not. The capsule layer is *partially* load-bearing in
  execution; the next slice would extend it.

### 4.2 What the post-hoc fold still does

``build_report_ledger`` continues to be public SDK surface. Use it
when:

- You have a ``product_report.json`` dict from a third party
  whose runtime did not emit a capsule view (legacy v2 reports
  or v3.0 reports without the in-flight extension).
- You are writing a *consumer* tool that wants to re-derive the
  capsule DAG from a finished report, e.g. a report-comparison
  CLI that loads two reports and compares their capsule graphs.
- You are testing — the post-hoc fold is the canonical
  reference implementation that the in-flight builder is
  CID-equivalent to (Theorem W3-34).

The post-hoc fold remains the *adapter*; the in-flight builder
is the *runtime*.

---

## 5. Status table (single page)

| Claim                              | Status                              |
|---                                 |---                                  |
| W3-32 lifecycle correspondence    | **Proved** by inspection + tested |
| W3-33 content-addressing at write | **Proved** by inspection + tested |
| W3-34 in-flight ↔ post-hoc CID equivalence (non-ARTIFACT) | **Proved** by per-CID test |
| W3-35 parent-CID gating          | **Proved** by inspection + tested |
| Post-hoc fold remains supported   | **Code-backed** (legacy path runs green) |
| Substantive artifacts content-addressed | **Code-backed** (test_substantive_artifacts_are_content_addressed) |
| Meta-artifacts content-addressed  | **Conjecturally not closeable** (circular dependency on view containing itself) |
| Intra-cell objects (prompt / parsed patch / verdict) capsule-native | **Open / next slice** |
| Re-execution-deterministic CIDs across runs (full DAG) | **Open / requires a deterministic-mode toggle** |
| Adversarial concurrent-writer detection | **Out of scope** |

---

## 6. Code anchors

- ``vision_mvp/wevra/capsule_runtime.py`` —
  ``CapsuleNativeRunContext`` (one ledger, eight stage methods),
  ``seal_and_write_artifact`` (free function), constants
  ``CONSTRUCTION_IN_FLIGHT`` / ``CONSTRUCTION_POST_HOC``,
  ``ContentAddressMismatch`` exception.
- ``vision_mvp/wevra/runtime.py`` — ``run_sweep`` accepts
  optional ``ctx`` and seals each cell via
  ``ctx.seal_sweep_cell`` in flight.
- ``vision_mvp/product/runner.py`` —
  ``_run_profile_capsule_native`` (the new default path) and
  ``_run_profile_post_hoc`` (the legacy path).
- ``vision_mvp/wevra/run.py`` — ``RunSpec.capsule_native`` flag
  (default True).
- ``vision_mvp/tests/test_wevra_capsule_native.py`` — 16 tests
  witnessing the four theorems.

Cross-references:
- The post-hoc fold's adapters
  (``vision_mvp/wevra/capsule.py::capsule_from_*``,
  ``build_report_ledger``) are unchanged byte-for-byte; the
  in-flight path *uses* them.
- The Capsule Formalism (``docs/CAPSULE_FORMALISM.md``)
  Theorems W3-1 through W3-31 are unchanged. SDK v3.1 adds
  W3-32 / W3-33 / W3-34 / W3-35 (this note).

---

*This results note is canonical for the SDK v3.1 milestone. If it
disagrees with any other file on what the capsule-native path
does, this note is right and the other file is stale.*
