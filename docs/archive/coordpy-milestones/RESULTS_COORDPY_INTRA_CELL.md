# RESULTS — CoordPy intra-cell capsule-native execution + detached witness (SDK v3.2)

> *Theory-forward results note. States the claim, the code anchors,
> the empirical status, the proof boundary, and what remains
> conjectural. Last touched: 2026-04-26.*

---

## 0. One-paragraph summary

SDK v3.1 closed the **run-boundary** slice: every cross-boundary
artefact between profile resolution and the run report became a
sealed capsule sealed *in flight*. Intra-cell objects (the patch a
generator emitted, the verdict the sandbox returned) still passed
between functions as plain Python dataclasses. SDK v3.2 makes the
next move: it **pushes capsules into the inner sweep loop** by
naming each (task, strategy) parse→apply→test transition as a pair
of typed lifecycle steps — ``PATCH_PROPOSAL`` (parent: SWEEP_SPEC)
and ``TEST_VERDICT`` (parent: PATCH_PROPOSAL) — sealed as the loop
runs. This pushes the capsule contract past the cell boundary into
the cell's interior. Two further moves complete the milestone:
(i) a **detached META_MANIFEST witness** seals the meta-artefact
boundary (Theorem W3-36 — the rendering circularity is sharp; the
strongest authentication is one explicit hop beyond the primary
view); (ii) **stronger on-disk verification** by ``coordpy-capsule
verify`` recomputes the chain from on-disk header bytes
(Theorem W3-37) and re-hashes every ARTIFACT and meta-artefact at
audit time (Theorem W3-38). The capsule contract is no longer just
a description of run boundaries; it now (a) gates intra-cell
apply→test ordering at the type level, (b) explicitly draws the
meta-artefact circularity boundary, and (c) verifiably re-checks
on-disk bytes against sealed CIDs.

---

## 1. The claim in one paragraph

> **The capsule-native slice is no longer just the run-boundary
> spine.** Inside every sweep cell, each (task, strategy) drives a
> pair of typed capsule lifecycle steps — a PATCH_PROPOSAL sealing
> the generator's output (parent: SWEEP_SPEC) and a TEST_VERDICT
> sealing the sandbox's apply+test outcome (parent: the
> PATCH_PROPOSAL whose patch was tested). The lifecycle ordering
> ``patch → verdict`` is enforced at the capsule layer; a verdict
> cannot be sealed before its patch (W3-35 extended to intra-cell
> kinds). The meta-artefact boundary (``product_report.json``,
> ``capsule_view.json``, ``product_summary.txt``) is now formally
> a **detached witness slice** outside the primary fixed point: a
> META_MANIFEST capsule, sealed in a *secondary* ledger after the
> RUN_REPORT, carries SHA-256s of the meta-artefacts plus the
> primary ``root_cid`` and ``chain_head``. ``coordpy-capsule verify``
> recomputes the primary chain from on-disk header bytes, re-hashes
> every ARTIFACT against the on-disk file, and cross-verifies the
> META_MANIFEST. Every step is by inspection of code that already
> ships.

---

## 2. New theorems

### 2.1 Intra-cell lifecycle correspondence (W3-32 extended)

**Setting.** Let one sweep cell admit a sequence of (task,
strategy) pairs $\{(\tau_i, \sigma_i)\}_{i=1}^N$. For each pair,
the runtime calls ``generator(task, ctx, ...)`` returning a
``ProposedPatch`` $p_i$, then ``sandbox.run(...)`` returning a
``WorkspaceResult`` $w_i$.

**Theorem W3-32-extended (intra-cell lifecycle correspondence).**
There exist two unique capsules per pair —
$c_{\rm patch,i} \in \mathcal{C}_\mathcal{R}$ of kind PATCH_PROPOSAL
with parent CID $\mathit{cid}(\mathrm{spec\_cap})$, and
$c_{\rm verdict,i} \in \mathcal{C}_\mathcal{R}$ of kind
TEST_VERDICT with parent CID $\mathit{cid}(c_{\rm patch,i})$ —
such that:

- **Patch in progress** at time $t$ iff $c_{\rm patch,i}$ has been
  proposed but not yet sealed;
- **Patch sealed** iff $c_{\rm patch,i}.\mathrm{cid} \in I(\mathcal{L})$;
- **Verdict in progress** iff $c_{\rm verdict,i}$ proposed but not
  sealed;
- **Verdict sealed** iff $c_{\rm verdict,i}.\mathrm{cid} \in I(\mathcal{L})$;
- A verdict cannot be sealed before its patch:
  ``ctx.seal_test_verdict`` raises ``CapsuleLifecycleError`` when
  $c_{\rm patch,i}.\mathrm{cid} \notin I(\mathcal{L})$.

**Proof.** ``CapsuleNativeRunContext.seal_patch_proposal``'s first
check is the SWEEP_SPEC sealing precondition, identical in shape
to ``seal_sweep_cell``'s check (W3-35). On success it constructs
the PATCH_PROPOSAL capsule with parent = $\mathit{cid}(\mathrm{spec\_cap})$
and goes through ``_propose`` → ``_admit_and_seal``, which is
the same primitive that drove W3-32. ``seal_test_verdict``'s
first check is ``patch_proposal_cid in self.ledger``; ledger
membership is ``cid in self._by_cid``, which is exactly the
parent-CID admissibility predicate (Capsule Formalism § 2.3, C5).
The two transitions form a chain: patch sealed → verdict
admissible → verdict sealed. $\square$

**Code anchor.**
``vision_mvp/coordpy/capsule_runtime.py``:
``CapsuleNativeRunContext.seal_patch_proposal`` and
``seal_test_verdict``. Hooks plumbed through
``vision_mvp/tasks/swe_sandbox.py::run_swe_loop_sandboxed``
(``on_patch_proposed`` and ``on_test_completed`` parameters,
default None preserves byte-for-byte behaviour).

**Empirical anchor.**
``vision_mvp/tests/test_coordpy_capsule_native_intra_cell.py``:
- ``test_patch_proposal_seals_under_spec`` (PATCH_PROPOSAL sealed
  with parent SWEEP_SPEC, payload is coordinates + content hash);
- ``test_patch_proposal_refuses_without_spec`` (lifecycle gate);
- ``test_test_verdict_seals_under_patch_proposal`` (TEST_VERDICT
  sealed with parent PATCH_PROPOSAL);
- ``test_test_verdict_refuses_without_patch`` (parent-CID gate);
- ``test_smoke_run_emits_intra_cell_capsules`` (end-to-end run on
  the bundled mock bank: 48 PATCH_PROPOSAL and 48 TEST_VERDICT
  capsules sealed in flight on local_smoke).

### 2.2 Meta-artefact circularity / detached witness (W3-36)

**Statement.** Let $R \in \mathcal{L}$ be the RUN_REPORT capsule of
a CoordPy run; let $V$ be the rendered view of the primary ledger at
$R$'s seal time (a function of the ledger's capsules and the
chain hash); let $M = \{m_1, m_2, m_3\}$ be the meta-artefact
set $\{\mathtt{product\_report.json},\, \mathtt{capsule\_view.json},
\,\mathtt{product\_summary.txt}\}$. The bytes of each $m_i$ are a
structural function of $V$ — ``product_report.json`` embeds $V$
literally as ``report["capsules"]``, ``capsule_view.json`` *is*
$V$, and ``product_summary.txt`` includes the by-kind histogram
and chain head from $V$.

**Theorem W3-36 (meta-artefact circularity is sharp).** There is no
extension $\mathcal{L}'$ of the primary ledger that admits an
ARTIFACT capsule $a_i$ with payload SHA-256
$\mathrm{SHA}\text{-}256(m_i)$ for $m_i \in M$ such that $a_i$'s
existence is encoded in $V'$ (the rendered view of $\mathcal{L}'$)
and $V' = V$.

**Proof.** Suppose toward contradiction that such $\mathcal{L}', a_i$
exist. Then $\mathcal{L}'$'s set of sealed capsules contains $a_i$
in addition to $\mathcal{L}$'s, so the rendered view $V'$ encodes
$a_i$ — either as a header in the ``capsules`` list or in the
chain-head computation (which depends on the ordered sequence of
sealed capsules' CIDs and kinds). Hence $V' \neq V$, contradicting
the assumption $V' = V$. $\square$

**Corollary (detached-witness existence).** The strongest
authentication of $M$ achievable is a *detached* witness: a
META_MANIFEST capsule sealed in a **secondary ledger**
$\mathcal{L}_{\rm meta}$ disjoint from $\mathcal{L}$, whose
payload is a function only of $(R.\mathrm{cid}, \mathcal{L}.\mathrm{chain\_head},
\{(m_i, \mathrm{SHA}\text{-}256(m_i), |m_i|)\}_{i})$. The
META_MANIFEST cross-references $\mathcal{L}$ through its payload's
``root_cid`` and ``chain_head`` fields but is not a capsule of
$\mathcal{L}$; it is the *one-hop trust unit* beyond the primary
view.

**Trust model.** A consumer holding the meta-artefacts plus the
META_MANIFEST trusts that the manifest itself is authentic (one
explicit hop). With that trust, the consumer can re-hash each
$m_i$ on disk and verify against the manifest's claim. Tampering
with $m_i$ is detected; tampering with the manifest itself is
detectable only if the consumer holds an out-of-band copy of
$R.\mathrm{cid}$.

**Code anchor.**
``vision_mvp/coordpy/capsule_runtime.py::CapsuleNativeRunContext.seal_meta_manifest``
and ``render_meta_manifest_view``;
``vision_mvp/coordpy/capsule.py::capsule_from_meta_manifest``;
``vision_mvp/product/runner.py`` Stage 8 (post-fixed-point) writes
``meta_manifest.json``.

**Empirical anchor.**
``vision_mvp/tests/test_coordpy_capsule_native_intra_cell.py``:
- ``test_meta_manifest_seals_in_secondary_ledger`` (manifest is
  not in the primary view; primary's ``construction`` field
  unchanged);
- ``test_meta_manifest_refuses_without_run_report`` (post-fixed-
  point gate);
- ``test_meta_manifest_shas_match_on_disk`` (every claimed SHA
  equals the actual on-disk SHA);
- ``test_w3_38_meta_manifest_drift_detected`` (tampering with a
  meta-artefact is observable through the manifest's verification).

### 2.3 Chain-from-headers verification (W3-37)

**Setting.** The capsule view's chain head $H$ is computed by the
runtime as $H = \mathrm{SHA\text{-}256}(\mathrm{prev}, \mathit{cid}, \mathrm{kind},
\mathtt{SEALED})$ folded over the sealed-capsule sequence. The
view's headers ``[{cid, kind, lifecycle, ...}, ...]`` carry the
three load-bearing fields per capsule — the lifecycle is forced
to ``SEALED`` in the chain step.

**Theorem W3-37 (chain-from-headers verification).** Let
``view_dict`` be a serialised capsule view (from
``capsule_view.json`` or ``product_report["capsules"]``). Define
``verify_chain_from_view_dict(view_dict)`` to recompute the chain
head from the on-disk headers. Then:

$$
\mathrm{verify\_chain\_from\_view\_dict}(\mathit{view\_dict})
= \mathit{True}
\;\;\iff\;\;
\mathrm{recompute}(\mathit{view\_dict}) = \mathit{view\_dict}[\mathtt{chain\_head}].
$$

**Proof.** The chain step is a pure function of (prev, cid, kind,
SEALED). All three fields are JSON-serialised in the on-disk
header dict (``{"cid": ..., "kind": ..., "lifecycle": ...}``). The
recomputation reproduces the runtime's exact ``_chain_step``
function over the on-disk bytes; the equality of the recomputed
head and the on-disk head is the verification predicate. $\square$

**Tamper detection.** Any of the following on-disk perturbations
flip the verification result to False:
(i) flipping a CID in the headers list;
(ii) flipping a kind;
(iii) reordering the capsules list;
(iv) rewriting the chain_head field directly;
(v) inserting / deleting a header.

**Code anchor.**
``vision_mvp/coordpy/capsule.py::verify_chain_from_view_dict``.

**Empirical anchor.**
``vision_mvp/tests/test_coordpy_capsule_native_intra_cell.py``:
``test_w3_37_chain_recompute_from_view``,
``test_w3_37_tamper_detected``,
``test_w3_37_tamper_capsule_order_detected``.

### 2.4 ARTIFACT bytes on disk re-hash (W3-38)

**Setting.** Theorem W3-33 establishes that
``seal_and_write_artifact``'s post-condition is
$\mathrm{SHA\text{-}256}(\mathrm{read}(p)) = c_A.\mathit{payload}[\texttt{"sha256"}]$
*at return time*. W3-33 is silent on **audit-time** drift: a file
edited after the runtime returns does not violate W3-33.

**Theorem W3-38 (ARTIFACT audit-time re-hash).** Let
``view_dict`` be a serialised view containing ARTIFACT capsules
with payload SHA-256 fields. Define
``verify_artifacts_on_disk(view_dict, base_dir)`` to re-read each
capsule's ``payload["path"]`` under ``base_dir`` and re-hash the
on-disk bytes. Then for every ARTIFACT capsule $c_A$ with a real
SHA in payload (i.e. produced by the in-flight path, not the
post-hoc fold), the verification reports OK iff the on-disk file
hashes to $c_A.\mathit{payload}[\texttt{"sha256"}]$.

**Proof.** By W3-33, at runtime return the on-disk SHA equals the
payload SHA. At audit time, the bytes on disk may have changed
(adversarial edit, accidental overwrite, FS corruption); the
audit-time SHA is the truthful reflection. The verification
returns OK iff the two equal. $\square$

**Code anchor.**
``vision_mvp/coordpy/capsule_runtime.py::verify_artifacts_on_disk``.

**Empirical anchor.**
``vision_mvp/tests/test_coordpy_capsule_native_intra_cell.py``:
``test_w3_38_artifact_on_disk_re_hash`` (clean run is OK);
``test_w3_38_artifact_drift_detected`` (a tampered substantive
artefact returns BAD with a drift entry naming the file).

---

## 3. What is NOW true about CoordPy

The set of statements that *became* true with SDK v3.2:

- **The capsule-native slice extends past the cell boundary into
  the inner sweep loop.** Each (task, strategy) parse→apply→test
  transition is sealed as a typed pair of capsules in the primary
  ledger, with parent-CID gating enforcing the patch→verdict
  ordering. Mid-cell failure is observable as a typed in-flight
  register entry.

- **The meta-artefact boundary is formally a circularity, with a
  sharp limitation theorem and a positive corollary.** The
  rendering of the RUN_REPORT-rooted view cannot be
  authenticated within the primary ledger (W3-36); the strongest
  authentication achievable is a detached META_MANIFEST in a
  secondary ledger (W3-36 corollary). The runtime now writes
  ``meta_manifest.json`` on every run.

- **``coordpy-capsule verify`` is no longer a header-trust check.**
  It recomputes the chain from on-disk header bytes (W3-37),
  re-hashes every ARTIFACT capsule's on-disk file (W3-38), and
  cross-verifies the detached META_MANIFEST. Each check is
  printed individually plus a final OK / BAD verdict; exit code
  3 on any failure.

- **The view always includes payloads for content-addressing
  kinds.** Under the pre-v3.2 default ``include_payload=False``,
  ARTIFACT and META_MANIFEST capsules now ALWAYS carry their
  payloads in the rendered view. The payloads carry the
  verification claim; without them the on-disk re-hash check
  cannot run. The footprint cost is small (a few hundred bytes
  per capsule).

- **Theorem W3-34's spine equivalence is preserved.** The
  intra-cell capsule extension is *additive*: PATCH_PROPOSAL and
  TEST_VERDICT capsules are siblings of SWEEP_CELL via SWEEP_SPEC,
  not modifications of the spine. The post-hoc
  ``build_report_ledger`` fold continues to produce CID-equivalent
  ledgers on PROFILE / READINESS_CHECK / SWEEP_SPEC / SWEEP_CELL /
  PROVENANCE.

The set of statements that are *not yet* true:

- **Generator prompts and parser outputs are capsule-tracked.**
  The intra-cell slice currently captures the *patch* (the
  parser's output dressed as a ``ProposedPatch``) and the
  *verdict* (the sandbox's apply+test outcome). The LLM prompt
  itself, the raw LLM response, and the parser's
  ``ParseOutcome`` (kind / recovery label / detail) are not yet
  capsule-tracked. Naming them as further intra-cell kinds
  (``PROMPT``, ``LLM_RESPONSE``, ``PARSE_OUTCOME``) is the next
  natural extension.

- **The detached META_MANIFEST is itself authenticated by the
  primary ledger.** Theorem W3-36 establishes this is *impossible*
  without circularity; the limitation is sharp, not a TODO.
  Cryptographic signing of the META_MANIFEST (e.g. with a
  per-run keypair) is the only way to push the trust unit
  further out, and that is out of scope.

- **Adversarial concurrent writers are detected.** W3-37 and
  W3-38 are honest-writer audits; an attacker with concurrent
  write access to the output directory is out of scope.

- **Cross-run RUN_REPORT determinism on the full DAG.** Two runs
  of the same profile produce different PROVENANCE / RUN_REPORT
  CIDs (timestamps); SDK v3.2 does not change this.

---

## 4. Honest limitations

### 4.1 What the intra-cell slice does NOT promise

- **All inner-loop bytes are capsule-tracked.** The patch and
  the verdict are; the LLM prompt and the raw LLM response and
  the parser's full taxonomy are not. The slice is a *real*
  step but not a *complete* one. An honest reading: SDK v3.2
  extends capsule-native execution past the cell boundary by
  one structural layer, not by all the way down.

- **Generator failure is fully attributable inside the
  capsule layer.** If the generator raises, the existing
  ``run_swe_loop_sandboxed`` substitutes ``ProposedPatch(patch=(),
  rationale="gen_error:...")`` and the loop continues — the
  PATCH_PROPOSAL capsule is sealed with an empty patch and the
  rationale records the error kind. This is loud, but it is not
  the same as naming the generator's exception as its own
  capsule kind.

- **The intra-cell ledger size scales with task count.** A
  100-instance bank × 5 strategies × 4 cells produces 2000
  PATCH_PROPOSAL and 2000 TEST_VERDICT capsules per run. RUN_REPORT
  intentionally does NOT include these as direct parents (the
  spine filter); the budget bump on RUN_REPORT (max_parents
  raised to $2^{16}$ in v3.2) is a safety, not the primary
  defence. Storage for very large sweeps is the user's concern;
  the capsule view will grow proportionally.

### 4.2 What the META_MANIFEST does NOT promise

- **A self-witnessing meta-artefact.** No file in the run can
  authenticate itself by re-encoding its own SHA. Theorem W3-36
  formalises this. The detached witness IS the answer, and the
  trust unit is one hop.

- **Cross-host portability without a side-channel.** A consumer
  who downloads an output directory needs the META_MANIFEST to
  authenticate the meta-artefacts. If the META_MANIFEST is
  shipped alongside the meta-artefacts, an attacker who tampers
  with both is undetectable from disk alone. Pushing the trust
  unit further requires an out-of-band signature or the
  primary ``root_cid`` shipped through a separate channel.

### 4.3 What strong verification does NOT promise

- **The primary ledger's content is honest.** W3-37 detects
  tampering of bytes; it does not validate semantics. A capsule
  sealed under a forged-but-self-consistent payload will pass
  every chain check.

- **Integrity over time without a witness.** The runtime
  computes one SHA; the audit recomputes one SHA. If the bytes
  change between, drift is detected. If the bytes change AND
  the in-memory CID is also rewritten in flight, the runtime
  would fail at ``ContentAddressMismatch`` (W3-33). The
  protections are layered, not infinite.

---

## 5. Status table (single page)

| Claim                                         | Status                              |
|---                                            |---                                  |
| W3-32-extended intra-cell lifecycle correspondence | **Proved** by inspection + 5 contract tests |
| W3-36 meta-artefact circularity (negative)        | **Proved** by structural argument |
| W3-36 detached-witness existence (positive corollary) | **Code-backed** + 4 contract tests |
| W3-37 chain-from-headers verification             | **Proved** by inspection + 3 contract tests (incl. tamper) |
| W3-38 ARTIFACT audit-time re-hash                 | **Proved** by inspection + 2 contract tests (incl. drift) |
| Theorem W3-34 spine equivalence preserved under v3.2 | **Code-backed** by re-run of the W3-34 spine-CID test under SDK v3.2 |
| Intra-cell extension to PROMPT / LLM_RESPONSE / PARSE_OUTCOME | **Open / next slice** |
| Cryptographically signed META_MANIFEST           | **Out of scope** (orthogonal axis) |
| Cross-run determinism on full DAG                 | **Open** (requires deterministic-mode toggle) |
| Adversarial concurrent-writer detection           | **Out of scope** (trust boundary is the same as CoordPy's sandbox) |

---

## 6. Code anchors

- ``vision_mvp/coordpy/capsule.py`` — three new
  ``CapsuleKind`` constants (PATCH_PROPOSAL, TEST_VERDICT,
  META_MANIFEST) with default budgets, three new adapters
  (``capsule_from_patch_proposal``, ``capsule_from_test_verdict``,
  ``capsule_from_meta_manifest``), and ``verify_chain_from_view_dict``
  (the chain-from-headers recompute).
- ``vision_mvp/coordpy/capsule_runtime.py`` — three new methods on
  ``CapsuleNativeRunContext`` (``seal_patch_proposal``,
  ``seal_test_verdict``, ``seal_meta_manifest``); two new free
  functions (``verify_artifacts_on_disk``,
  ``verify_meta_manifest_on_disk``); ``seal_run_report``'s parent
  set restricted to spine kinds; ``render_view`` always carries
  ARTIFACT / META_MANIFEST / PATCH_PROPOSAL / TEST_VERDICT
  payloads.
- ``vision_mvp/coordpy/runtime.py`` — ``_make_intra_cell_hooks``
  builds the ``(on_patch_proposed, on_test_completed)`` closure
  pair that the unified mock + real sweep loops pass into the
  sandbox-aware substrate.
- ``vision_mvp/tasks/swe_sandbox.py`` —
  ``run_swe_loop_sandboxed`` accepts ``on_patch_proposed`` /
  ``on_test_completed`` (default None preserves byte-for-byte
  Phase-40 behaviour); the inner loop now calls both hooks at
  the right moments.
- ``vision_mvp/product/runner.py`` —
  ``_run_profile_capsule_native`` Stage 8 (post-fixed-point)
  computes meta-artefact SHAs, seals the META_MANIFEST in the
  secondary ledger, writes ``meta_manifest.json``.
- ``vision_mvp/coordpy/_cli.py`` — ``coordpy-capsule verify`` now
  runs four independent on-disk checks (chain recompute embedded,
  chain recompute on-disk, ARTIFACT bytes, META_MANIFEST), prints
  each line, and emits exit code 3 on any failure.
- ``vision_mvp/tests/test_coordpy_capsule_native_intra_cell.py`` —
  16 contract tests locking the four new theorems and the W3-34
  spine equivalence preservation.

Cross-references unchanged byte-for-byte:
- The post-hoc fold's adapters (``capsule_from_*`` for spine
  kinds) and ``build_report_ledger`` are untouched. The post-hoc
  path under ``capsule_native=False`` is byte-for-byte the same
  as in v3.1.
- The Capsule Formalism's W3-1 through W3-31 are unchanged.
  v3.2 adds W3-36, W3-37, W3-38 and extends W3-32 to intra-cell
  kinds.
- The Capsule Contract C1..C6 is unchanged.

---

*This results note is canonical for the SDK v3.2 milestone. If it
disagrees with any other file on what the capsule-native path does
inside the cell, this note is right and the other file is stale.*
