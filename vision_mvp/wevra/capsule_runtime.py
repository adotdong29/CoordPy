"""Capsule-native runtime — capsules drive execution, not just audit.

Up to SDK v3, the capsule layer was a *post-hoc* fold: a Wevra run
ran end-to-end as ordinary Python, the resulting ``product_report``
dict was handed to ``build_report_ledger``, and a sealed capsule
DAG was synthesised after the fact. Capsules described the run;
they did not gate it.

This module makes the next move. ``CapsuleNativeRunContext`` is a
runtime context that *owns* a ``CapsuleLedger`` and exposes the
exact lifecycle transitions the run must traverse. Each runtime
stage — profile resolution, readiness check, sweep spec admission,
per-cell execution, provenance manifest, artifact emission, and
the final run-report — corresponds to a capsule transition that
happens *in flight*. A stage that fails leaves an in-flight
``PROPOSED`` entry that never reaches ``SEALED``; downstream
stages refuse to start because the parent CID is missing from
the ledger (Capsule Contract invariant C5). Mid-run failure is
therefore a *typed* observation about which capsule never sealed,
not a bag-of-state-bytes.

Two orthogonal moves are coupled here:

  1.  **Lifecycle correspondence (Theorem W3-32).** A bijection
      between the runtime's execution-state machine and capsule
      lifecycle states. A stage S is in progress at time t iff its
      capsule c is in the in-flight register but not in the ledger;
      S has completed at t iff c is SEALED in the ledger; S failed
      iff c stays in the in-flight register and never seals.

  2.  **Content-addressing at write time (Theorem W3-33).** An
      ``ARTIFACT`` capsule's SHA-256 is computed from the bytes
      *before* they hit disk; the capsule is sealed under that
      hash; the bytes are then committed, re-read, and re-hashed
      to verify the post-condition ``SHA-256(read(path)) ==
      sealed_cap.payload["sha256"]``. The on-disk file is
      authenticated by its capsule, not by trust in the writer.

What this module is NOT
-----------------------

  * Not a replacement for ``build_report_ledger``. The post-hoc
    fold is retained for third parties who want to lift a finished
    ``product_report`` dict (e.g. a report imported from disk into
    a different process) into a capsule DAG. ``build_report_ledger``
    is a deterministic adapter; ``CapsuleNativeRunContext`` is a
    runtime owner. Both produce CID-equivalent ledgers for the
    non-artifact kinds (Theorem W3-34); they differ only in the
    ARTIFACT kind, where the runtime path includes real SHA-256
    hashes and the post-hoc fold leaves them ``None``.

  * Not a rewrite of the readiness, sweep, sandbox, or LLM-client
    primitives. The legacy code paths under ``vision_mvp.tasks.*``,
    ``vision_mvp.experiments.phase44_public_readiness``, and the
    ``run_swe_loop_sandboxed`` execution loop run byte-for-byte the
    same. What changes is the *contract* under which their outputs
    cross the run-boundary: every output now becomes a sealed
    capsule before the next stage can read it.

  * Not a guarantee on adversarial concurrency. The post-write
    re-hash check is a TOCTOU detector for honest writers, not a
    defence against an adversary with concurrent write access to
    the output directory. The trust boundary is the same as Wevra's
    sandbox boundary (see ``vision_mvp.tasks.swe_sandbox``).

Public surface (additive on SDK v3)
-----------------------------------

  * ``CapsuleNativeRunContext`` — the runtime context.
  * ``seal_and_write_artifact`` — content-addressed artifact writer
    (a free function for callers who don't want the full context).
  * ``CONSTRUCTION_IN_FLIGHT`` / ``CONSTRUCTION_POST_HOC`` — string
    constants tagged into the rendered ``CapsuleView`` so consumers
    can tell which path produced the ledger.
  * ``ContentAddressMismatch`` — raised when a post-write re-hash
    detects on-disk bytes drift from the sealed CID.

The module is intentionally small and reads as a state machine.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from typing import Any, Iterable

from .capsule import (
    CAPSULE_VIEW_SCHEMA,
    CapsuleAdmissionError,
    CapsuleBudget,
    CapsuleKind,
    CapsuleLedger,
    CapsuleLifecycle,
    CapsuleLifecycleError,
    CapsuleView,
    ContextCapsule,
    _default_budget_for,
    capsule_from_artifact,
    capsule_from_meta_manifest,
    capsule_from_patch_proposal,
    capsule_from_profile,
    capsule_from_provenance,
    capsule_from_readiness,
    capsule_from_report,
    capsule_from_sweep_cell,
    capsule_from_sweep_spec,
    capsule_from_test_verdict,
    render_view,
)


# Construction-mode tags. Carried alongside ``wevra.capsule_view.v1``
# so a downstream consumer can tell whether the ledger was built in
# flight (capsules drove execution) or post hoc (capsules described a
# finished report). Both are valid; only the first earns the
# "execution contract" claim.
CONSTRUCTION_IN_FLIGHT = "in_flight"
CONSTRUCTION_POST_HOC = "post_hoc"


class ContentAddressMismatch(CapsuleAdmissionError):
    """Raised by ``seal_and_write_artifact`` when the on-disk
    file's SHA-256 does not equal the SHA-256 in the sealed
    ARTIFACT capsule's payload.

    Inherits from ``CapsuleAdmissionError`` because it is the
    runtime form of a Contract C1 violation: the capsule's CID was
    derived under the in-memory bytes but the on-disk artifact does
    not match. Either the writer is buggy, the filesystem corrupted
    the bytes, or another process wrote over the file between
    ``seal`` and the post-write re-hash.
    """


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclasses.dataclass
class _InFlightEntry:
    """One entry in the runtime's *transient* register.

    The capsule ledger only stores SEALED capsules. A PROPOSED
    capsule that has not yet been admitted lives in the in-flight
    register only; if a stage fails, the capsule never reaches the
    ledger and its in-flight entry remains as the typed witness of
    the failure.
    """

    kind: str
    capsule: ContextCapsule
    proposed_at: float
    sealed_at: float | None = None
    sealed_cid: str | None = None
    failure: str | None = None

    def status(self) -> str:
        if self.failure:
            return "failed"
        if self.sealed_cid:
            return "sealed"
        return "in_flight"


class CapsuleNativeRunContext:
    """Runtime owner of a ``CapsuleLedger`` for one Wevra run.

    Lifecycle (one canonical run):

      1. ``start_run(profile_name, profile_dict)`` — seals the
         ``PROFILE`` capsule. Required first. Other stages refuse
         to seal until profile is sealed.
      2. ``seal_readiness(verdict)`` — seals a ``READINESS_CHECK``
         capsule with parent = profile.
      3. ``seal_sweep_spec(spec_payload)`` — seals a ``SWEEP_SPEC``
         capsule with parent = profile. Cells refuse to seal
         without it.
      4. ``seal_sweep_cell(cell_payload)`` — seals one
         ``SWEEP_CELL`` capsule per executed cell, with parent =
         sweep_spec. Idempotent on payload.
      5. ``seal_provenance(manifest)`` — seals a ``PROVENANCE``
         capsule with parent = profile.
      6. ``seal_and_write_artifact(path, data, parents=...)`` —
         seals an ``ARTIFACT`` capsule whose payload carries the
         SHA-256 of ``data``, then writes ``data`` to ``path``,
         then re-hashes and verifies. Raises
         ``ContentAddressMismatch`` on drift.
      7. ``seal_run_report(headers)`` — seals the ``RUN_REPORT``
         capsule with parents = every other sealed capsule. The
         RUN_REPORT capsule's CID is the run's durable identifier.
      8. ``render`` — return a ``CapsuleView`` tagged with
         ``construction=in_flight``.

    Every step admits and seals; intermediate ``PROPOSED`` /
    ``ADMITTED`` states only exist between the construction of the
    capsule and the call to ``admit_and_seal``. The in-flight
    register tracks failures: a step that raises will leave a
    ``failed`` entry that never reaches the ledger.
    """

    def __init__(self) -> None:
        self.ledger: CapsuleLedger = CapsuleLedger()
        self.profile_cap: ContextCapsule | None = None
        self.readiness_cap: ContextCapsule | None = None
        self.spec_cap: ContextCapsule | None = None
        self.cell_caps: list[ContextCapsule] = []
        self.provenance_cap: ContextCapsule | None = None
        self.artifact_caps: list[ContextCapsule] = []
        self.run_report_cap: ContextCapsule | None = None
        # SDK v3.2 — intra-cell capsule-native slice. ``patch_caps``
        # and ``verdict_caps`` accumulate as the inner sweep loop
        # seals one capsule per (task, strategy) for parse/apply/test
        # transitions. They live in the same primary ledger; their
        # parent chain links each TEST_VERDICT back to the
        # PATCH_PROPOSAL whose patch was tested, and each
        # PATCH_PROPOSAL back to the SWEEP_SPEC under which its
        # cell ran.
        self.patch_caps: list[ContextCapsule] = []
        self.verdict_caps: list[ContextCapsule] = []
        # SDK v3.2 — detached witness (Theorem W3-36). The
        # META_MANIFEST is sealed in a SECONDARY ledger after the
        # primary RUN_REPORT is sealed; it carries SHAs of the
        # meta-artifacts (product_report.json / capsule_view.json /
        # product_summary.txt) and the run's root_cid. It cannot
        # live in the primary ledger because adding it would
        # require the rendered view (already on disk) to encode
        # the new capsule — circular.
        self.meta_manifest_ledger: CapsuleLedger | None = None
        self.meta_manifest_cap: ContextCapsule | None = None
        self._in_flight: list[_InFlightEntry] = []
        self._started_at: float = time.time()

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _propose(self, cap: ContextCapsule) -> _InFlightEntry:
        entry = _InFlightEntry(
            kind=cap.kind, capsule=cap, proposed_at=time.time())
        self._in_flight.append(entry)
        return entry

    def _admit_and_seal(self, entry: _InFlightEntry) -> ContextCapsule:
        try:
            sealed = self.ledger.admit_and_seal(entry.capsule)
        except (CapsuleAdmissionError, CapsuleLifecycleError) as ex:
            entry.failure = f"{type(ex).__name__}: {ex}"
            raise
        entry.sealed_cid = sealed.cid
        entry.sealed_at = time.time()
        return sealed

    def _require_profile(self) -> ContextCapsule:
        if self.profile_cap is None:
            raise CapsuleLifecycleError(
                "no PROFILE capsule sealed; call start_run() first")
        return self.profile_cap

    # ------------------------------------------------------------
    # Stage 1: profile
    # ------------------------------------------------------------

    def start_run(self,
                  *,
                  profile_name: str,
                  profile_dict: dict[str, Any] | None,
                  ) -> ContextCapsule:
        """Seal the PROFILE capsule. Idempotent; calling twice
        returns the existing sealed capsule.

        The profile capsule is the root of the run's capsule DAG:
        every other capsule eventually points back to it through
        the parent chain.
        """
        if self.profile_cap is not None:
            return self.profile_cap
        cap = capsule_from_profile(profile_name, profile_dict)
        entry = self._propose(cap)
        self.profile_cap = self._admit_and_seal(entry)
        return self.profile_cap

    # ------------------------------------------------------------
    # Stage 2: readiness
    # ------------------------------------------------------------

    def seal_readiness(self,
                        verdict: dict[str, Any]) -> ContextCapsule:
        """Seal a READINESS_CHECK capsule with parent = profile.

        ``verdict`` is the dict returned by
        ``vision_mvp.experiments.phase44_public_readiness.run_readiness``.
        """
        prof = self._require_profile()
        cap = capsule_from_readiness(verdict, parents=(prof.cid,))
        entry = self._propose(cap)
        self.readiness_cap = self._admit_and_seal(entry)
        return self.readiness_cap

    # ------------------------------------------------------------
    # Stage 3: sweep spec
    # ------------------------------------------------------------

    def seal_sweep_spec(self,
                        spec_payload: dict[str, Any]) -> ContextCapsule:
        """Seal a SWEEP_SPEC capsule with parent = profile.

        ``spec_payload`` is the same shape ``build_report_ledger``
        uses (mode / sandbox / jsonl / model / endpoint /
        executed_in_process). This deliberate alignment makes the
        in-flight ledger CID-equivalent to a post-hoc fold of the
        same run for the SWEEP_SPEC kind (Theorem W3-34).
        """
        prof = self._require_profile()
        cap = capsule_from_sweep_spec(
            spec_payload, parents=(prof.cid,))
        entry = self._propose(cap)
        self.spec_cap = self._admit_and_seal(entry)
        return self.spec_cap

    # ------------------------------------------------------------
    # Stage 4: sweep cells (one per executed cell)
    # ------------------------------------------------------------

    def seal_sweep_cell(self,
                         cell_payload: dict[str, Any],
                         ) -> ContextCapsule:
        """Seal one SWEEP_CELL capsule with parent = sweep_spec.

        Refuses to seal if no SWEEP_SPEC has been sealed yet — the
        runtime's "no cells before spec" rule is enforced at the
        capsule layer, not by Python ordering convention.
        """
        if self.spec_cap is None:
            raise CapsuleLifecycleError(
                "no SWEEP_SPEC capsule sealed; call "
                "seal_sweep_spec() before seal_sweep_cell()")
        cap = capsule_from_sweep_cell(
            cell_payload, spec_cid=self.spec_cap.cid)
        entry = self._propose(cap)
        sealed = self._admit_and_seal(entry)
        self.cell_caps.append(sealed)
        return sealed

    # ------------------------------------------------------------
    # Stage 4b: intra-cell capsules (SDK v3.2)
    # ------------------------------------------------------------

    def seal_patch_proposal(self,
                              *,
                              instance_id: str,
                              strategy: str,
                              parser_mode: str,
                              apply_mode: str,
                              n_distractors: int,
                              substitutions,
                              rationale: str = "",
                              ) -> ContextCapsule:
        """Seal one ``PATCH_PROPOSAL`` capsule for a single (task,
        strategy) pair in the currently-running sweep cell.

        Parent: the sealed ``SWEEP_SPEC`` capsule. Refusing to seal
        without a SWEEP_SPEC is the same lifecycle gate that protects
        SWEEP_CELL (W3-35) — a patch proposal outside a sealed spec
        is meaningless.

        The capsule's payload is *coordinates + content hash* (see
        ``capsule_from_patch_proposal``); the full substitution
        bytes are not stored in the capsule (SHA is enough for the
        identity claim and a downstream consumer can re-hash a
        candidate patch byte-for-byte).
        """
        if self.spec_cap is None:
            raise CapsuleLifecycleError(
                "no SWEEP_SPEC capsule sealed; call "
                "seal_sweep_spec() before seal_patch_proposal() — "
                "intra-cell transitions cannot precede their cell "
                "spec (Theorem W3-35 extended to intra-cell kinds)")
        cap = capsule_from_patch_proposal(
            instance_id=instance_id, strategy=strategy,
            parser_mode=parser_mode, apply_mode=apply_mode,
            n_distractors=n_distractors,
            substitutions=substitutions, rationale=rationale,
            parents=(self.spec_cap.cid,))
        entry = self._propose(cap)
        sealed = self._admit_and_seal(entry)
        self.patch_caps.append(sealed)
        return sealed

    def seal_test_verdict(self,
                            *,
                            instance_id: str,
                            strategy: str,
                            parser_mode: str,
                            apply_mode: str,
                            n_distractors: int,
                            patch_proposal_cid: str,
                            patch_applied: bool,
                            syntax_ok: bool,
                            test_passed: bool,
                            error_kind: str = "",
                            error_detail: str = "",
                            ) -> ContextCapsule:
        """Seal one ``TEST_VERDICT`` capsule for a single (task,
        strategy) pair, parented on the upstream PATCH_PROPOSAL
        whose patch was tested.

        Refusing to seal if ``patch_proposal_cid`` is not in the
        ledger is the parent-CID gate from Capsule Contract C5: a
        verdict cannot precede the patch it tests. This is W3-35's
        rule extended to intra-cell kinds — the lifecycle ordering
        ``patch → verdict`` is enforced at the type level.
        """
        if patch_proposal_cid not in self.ledger:
            raise CapsuleLifecycleError(
                f"no PATCH_PROPOSAL capsule with CID "
                f"{patch_proposal_cid[:12]}… is sealed; cannot seal "
                f"TEST_VERDICT before its patch (intra-cell "
                f"lifecycle gate, Theorem W3-35 extended)")
        cap = capsule_from_test_verdict(
            instance_id=instance_id, strategy=strategy,
            parser_mode=parser_mode, apply_mode=apply_mode,
            n_distractors=n_distractors,
            patch_applied=patch_applied,
            syntax_ok=syntax_ok,
            test_passed=test_passed,
            error_kind=error_kind,
            error_detail=error_detail,
            parents=(patch_proposal_cid,))
        entry = self._propose(cap)
        sealed = self._admit_and_seal(entry)
        self.verdict_caps.append(sealed)
        return sealed

    # ------------------------------------------------------------
    # Stage 5: provenance
    # ------------------------------------------------------------

    def seal_provenance(self,
                         manifest: dict[str, Any],
                         ) -> ContextCapsule:
        """Seal a PROVENANCE capsule with parent = profile."""
        prof = self._require_profile()
        cap = capsule_from_provenance(
            manifest, parents=(prof.cid,))
        entry = self._propose(cap)
        self.provenance_cap = self._admit_and_seal(entry)
        return self.provenance_cap

    # ------------------------------------------------------------
    # Stage 6: artifact emission (content-addressed at write time)
    # ------------------------------------------------------------

    def seal_and_write_artifact(self,
                                 *,
                                 path: str,
                                 data: bytes,
                                 parents: Iterable[str] | None = None,
                                 ) -> ContextCapsule:
        """Content-addressed artifact emission.

        Order of operations (Theorem W3-33):
          1. Compute SHA-256 of ``data`` in memory.
          2. Build an ARTIFACT capsule with payload
             ``{"path": path, "sha256": <hex>}``.
          3. Admit + seal in the ledger. If admission fails (over
             budget, missing parent, etc.) NOTHING is written to
             disk — the in-flight entry records the failure.
          4. Write ``data`` to ``path`` (creating parent dirs).
          5. Re-read the on-disk file and re-hash. If the hash
             differs from the sealed CID's payload SHA, raise
             ``ContentAddressMismatch``.

        The post-write re-hash is a TOCTOU detector for honest
        writers: if the bytes are stable between seal and re-read,
        the on-disk file is *authenticated by its capsule's CID*.
        Anyone holding the capsule's CID and the bytes can verify
        the bytes are the bytes the run produced.

        ``parents`` defaults to ``(profile_cap.cid,)`` if the
        profile has been sealed, matching the post-hoc fold's
        ARTIFACT structure. Callers may override with a tighter
        parent set (e.g. the readiness capsule's CID for the
        readiness_verdict.json artifact).
        """
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                f"seal_and_write_artifact requires bytes, "
                f"got {type(data).__name__}")
        sha = _sha256_bytes(bytes(data))
        if parents is None:
            parents = ((self.profile_cap.cid,)
                       if self.profile_cap else ())
        cap = capsule_from_artifact(
            path, sha256=sha, parents=tuple(parents))
        entry = self._propose(cap)
        sealed = self._admit_and_seal(entry)
        # Only AFTER the capsule is sealed do we commit bytes to
        # disk. If admission failed above, no file was written.
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(bytes(data))
        # Post-write re-hash: detects TOCTOU drift / corrupting
        # writers / racing processes.
        actual = _sha256_path(path)
        if actual != sha:
            raise ContentAddressMismatch(
                f"on-disk SHA mismatch for {path!r}: "
                f"sealed_cid_sha256={sha[:16]}… "
                f"on_disk_sha256={actual[:16]}…")
        self.artifact_caps.append(sealed)
        return sealed

    def seal_artifact_from_path(self,
                                 path: str,
                                 *,
                                 parents: Iterable[str] | None = None,
                                 ) -> ContextCapsule:
        """Variant: hash a file already on disk and seal an
        ARTIFACT capsule for it.

        Useful for files written by legacy code paths that the
        runner does not control (e.g. artifacts produced by
        ``run_swe_loop_sandboxed``'s subprocess sandboxes when the
        bytes never pass through the Python parent process). The
        capsule's payload SHA matches the on-disk file at the
        moment of the call; subsequent writes to the path are
        detectable as a CID mismatch.
        """
        if parents is None:
            parents = ((self.profile_cap.cid,)
                       if self.profile_cap else ())
        sha = _sha256_path(path)
        cap = capsule_from_artifact(
            path, sha256=sha, parents=tuple(parents))
        entry = self._propose(cap)
        sealed = self._admit_and_seal(entry)
        self.artifact_caps.append(sealed)
        return sealed

    # ------------------------------------------------------------
    # Stage 7: run report
    # ------------------------------------------------------------

    # The run-boundary spine — the kinds whose CIDs are direct
    # parents of RUN_REPORT. Intra-cell kinds (PATCH_PROPOSAL,
    # TEST_VERDICT) are deliberately excluded: they live in the
    # primary ledger as siblings of SWEEP_CELL, but RUN_REPORT
    # references the *spine* of the run (the run-boundary
    # crossings), not every transient capsule. This keeps RUN_REPORT
    # parent-set CID-equivalent with ``build_report_ledger``'s
    # post-hoc fold on the spine kinds (Theorem W3-34 preserved
    # under the SDK v3.2 intra-cell extension).
    _SPINE_KINDS_FOR_RUN_REPORT: frozenset[str] = frozenset({
        CapsuleKind.PROFILE,
        CapsuleKind.READINESS_CHECK,
        CapsuleKind.SWEEP_SPEC,
        CapsuleKind.SWEEP_CELL,
        CapsuleKind.PROVENANCE,
        CapsuleKind.ARTIFACT,
    })

    def seal_run_report(self,
                         headers: dict[str, Any],
                         ) -> ContextCapsule:
        """Seal the RUN_REPORT capsule.

        Parents are every sealed *spine* capsule in the ledger so
        far — PROFILE, READINESS_CHECK, SWEEP_SPEC, SWEEP_CELL,
        PROVENANCE, ARTIFACT. Intra-cell capsules (PATCH_PROPOSAL,
        TEST_VERDICT, introduced in SDK v3.2) are deliberately not
        parents of RUN_REPORT — they hang off SWEEP_SPEC and form
        a *transitively reachable* sub-DAG, but RUN_REPORT names
        only the run-boundary spine. This (a) preserves Theorem
        W3-34's spine equivalence with the post-hoc fold and
        (b) keeps the RUN_REPORT parent-set bounded by the spine
        even on huge sweeps.

        The run-report capsule's CID is the durable identifier
        for the run.
        """
        parent_cids = tuple(
            c.cid for c in self.ledger.all_capsules()
            if c.kind in self._SPINE_KINDS_FOR_RUN_REPORT)
        budget = _default_budget_for(CapsuleKind.RUN_REPORT)
        if (budget.max_parents is not None
                and len(parent_cids) > budget.max_parents):
            parent_cids = parent_cids[: budget.max_parents]
        cap = capsule_from_report(headers, parents=parent_cids)
        entry = self._propose(cap)
        self.run_report_cap = self._admit_and_seal(entry)
        return self.run_report_cap

    # ------------------------------------------------------------
    # Stage 8 (post-fixed-point): meta-manifest detached witness
    # ------------------------------------------------------------

    def seal_meta_manifest(self,
                            *,
                            meta_artifacts: list[dict[str, Any]],
                            ) -> ContextCapsule:
        """Seal a ``META_MANIFEST`` capsule in a *secondary* ledger
        as the detached witness for meta-artifacts.

        Theorem W3-36 (Detached-witness theorem). The set of
        meta-artifacts whose bytes are a structural function of
        the rendered RUN_REPORT view (``product_report.json``,
        ``capsule_view.json``, ``product_summary.txt``) cannot be
        authenticated by ARTIFACT capsules in the primary ledger
        without circular dependence: their bytes already encode
        the rendered view, so any new ARTIFACT capsule whose
        payload is their SHA must be present in the rendered view
        — but the rendered view is what the meta-artifacts
        encode. The strongest authentication achievable is a
        detached META_MANIFEST sealed *after* the RUN_REPORT in a
        secondary ledger; the META_MANIFEST is the one-hop trust
        unit beyond the primary view.

        Implementation: the manifest lives in
        ``self.meta_manifest_ledger`` (a fresh ``CapsuleLedger``),
        not the primary ``self.ledger``. The manifest's parent is
        the RUN_REPORT capsule's CID — but recorded as a *named*
        link, not as a parent that the secondary ledger would
        admission-check (the secondary ledger does not contain
        RUN_REPORT). The link is documented in payload's
        ``root_cid`` field.

        ``meta_artifacts`` is a list of
        ``{"path": str, "sha256": str, "n_bytes": int}`` dicts.

        Refuses to seal if RUN_REPORT has not yet been sealed —
        the manifest is by definition post-fixed-point.
        """
        if self.run_report_cap is None:
            raise CapsuleLifecycleError(
                "no RUN_REPORT capsule sealed; the META_MANIFEST is "
                "a post-fixed-point witness and cannot precede the "
                "primary ledger's root (Theorem W3-36)")
        # Secondary ledger — independent of the primary. The
        # secondary's chain is a singleton (one capsule). Anyone
        # holding both ledgers can audit each independently; the
        # cross-reference is the ``root_cid`` field in the
        # manifest's payload.
        self.meta_manifest_ledger = CapsuleLedger()
        cap = capsule_from_meta_manifest(
            root_cid=self.run_report_cap.cid,
            chain_head=self.ledger.chain_head(),
            meta_artifacts=meta_artifacts,
            parents=(),  # detached: secondary ledger, no in-ledger
                          # parents (the cross-reference is the
                          # root_cid in the payload).
        )
        sealed = self.meta_manifest_ledger.admit_and_seal(cap)
        self.meta_manifest_cap = sealed
        return sealed

    def render_meta_manifest_view(self) -> dict[str, Any]:
        """Render the secondary (META_MANIFEST) ledger as a JSON-
        safe dict suitable for serialisation as
        ``meta_manifest.json``.

        Returns ``None`` if the manifest has not been sealed.
        """
        if (self.meta_manifest_ledger is None
                or self.meta_manifest_cap is None):
            return None
        view = render_view(
            self.meta_manifest_ledger,
            include_payload=True,
            root_cid=self.meta_manifest_cap.cid,
        ).as_dict()
        # Tag as detached so consumers can tell at a glance.
        view["construction"] = "detached_witness"
        view["primary_root_cid"] = (
            self.run_report_cap.cid if self.run_report_cap else None)
        view["primary_chain_head"] = self.ledger.chain_head()
        return view

    # ------------------------------------------------------------
    # Rendering & introspection
    # ------------------------------------------------------------

    def render(self,
                *, include_payload: bool = False,
                ) -> dict[str, Any]:
        """Render the in-flight ledger as a dict suitable for
        embedding under ``product_report["capsules"]``.

        The returned dict matches the shape of
        ``CapsuleView.as_dict()`` and adds two additive fields:

          * ``"construction"``: ``"in_flight"`` — declares this
            view came from a runtime that drove execution through
            the capsule lifecycle.
          * ``"in_flight_stats"``: counts of proposed / sealed /
            failed entries observed during the run.

        These are additive on schema ``wevra.capsule_view.v1``;
        pre-v3 consumers ignore unknown keys, so the schema name
        is unchanged.
        """
        view = render_view(
            self.ledger,
            include_payload=include_payload,
            root_cid=(self.run_report_cap.cid
                       if self.run_report_cap else None),
        ).as_dict()
        view["construction"] = CONSTRUCTION_IN_FLIGHT
        view["in_flight_stats"] = self.in_flight_stats()
        return view

    def in_flight_stats(self) -> dict[str, int]:
        """Counts of in-flight register entries by status. The
        ``failed`` count is the number of capsules that were
        proposed but never sealed (admission rejection / lifecycle
        error). On a clean run, ``failed == 0`` and
        ``sealed == n_proposed``."""
        n_proposed = len(self._in_flight)
        n_sealed = sum(1 for e in self._in_flight if e.sealed_cid)
        n_failed = sum(1 for e in self._in_flight if e.failure)
        return {
            "n_proposed": n_proposed,
            "n_sealed": n_sealed,
            "n_failed": n_failed,
            "n_in_flight": n_proposed - n_sealed - n_failed,
        }

    def in_flight_failures(self) -> list[dict[str, Any]]:
        """List of dicts describing every PROPOSED-but-never-sealed
        capsule (one per stage that raised on admission). The
        runtime's typed witness of which stage(s) failed."""
        return [
            {
                "kind": e.kind,
                "cid": e.capsule.cid,
                "proposed_at": round(e.proposed_at, 3),
                "failure": e.failure,
            }
            for e in self._in_flight
            if e.failure
        ]


# =============================================================================
# Free-function content-addressed writer
# =============================================================================


def seal_and_write_artifact(ledger: CapsuleLedger,
                             *,
                             path: str,
                             data: bytes,
                             parents: Iterable[str] = (),
                             ) -> ContextCapsule:
    """Free-function form for callers who hold a ``CapsuleLedger``
    directly without a full ``CapsuleNativeRunContext``.

    Same contract as ``CapsuleNativeRunContext.seal_and_write_artifact``:
    seal first, then write, then verify.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(
            f"seal_and_write_artifact requires bytes, "
            f"got {type(data).__name__}")
    sha = _sha256_bytes(bytes(data))
    cap = capsule_from_artifact(path, sha256=sha, parents=tuple(parents))
    sealed = ledger.admit_and_seal(cap)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(bytes(data))
    actual = _sha256_path(path)
    if actual != sha:
        raise ContentAddressMismatch(
            f"on-disk SHA mismatch for {path!r}: "
            f"sealed_cid_sha256={sha[:16]}… "
            f"on_disk_sha256={actual[:16]}…")
    return sealed


__all__ = [
    "CapsuleNativeRunContext",
    "ContentAddressMismatch",
    "seal_and_write_artifact",
    "CONSTRUCTION_IN_FLIGHT",
    "CONSTRUCTION_POST_HOC",
]


# =============================================================================
# Strong on-disk verification (SDK v3.2)
# =============================================================================


def verify_artifacts_on_disk(view: dict[str, Any],
                              *, base_dir: str | None = None,
                              ) -> dict[str, Any]:
    """Re-hash every ARTIFACT capsule's on-disk file and verify
    it matches the SHA-256 in the capsule's payload.

    This is the *strong* form of artifact verification: the
    runtime computes one SHA at write time (Theorem W3-33), and
    this function recomputes the SHA at audit time from the bytes
    that landed on disk. A drift between the two implies tamper /
    corruption / rewrite after seal.

    The ``view`` dict is the rendered capsule view (from
    ``product_report.json``'s ``capsules`` block or
    ``capsule_view.json``). Under SDK v3.2, ARTIFACT capsule
    payloads are *always* included in the view — see
    ``render_view``'s docstring — so this function does not need
    a separate payload source.

    Returns a dict with:

      * ``verdict`` — overall "OK" / "BAD" / "PARTIAL"
      * ``checked``  — number of ARTIFACT capsules examined
      * ``ok``       — number whose on-disk SHA matched
      * ``mismatch`` — list of ``{path, sealed_sha, on_disk_sha}``
        for any drift
      * ``missing``  — list of paths declared in the view but not
        present on disk

    The ``base_dir`` argument is the directory where the on-disk
    files live; if ``None``, paths are interpreted as absolute /
    cwd-relative.
    """
    checked = 0
    ok = 0
    mismatch: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for cap in view.get("capsules", []):
        if cap.get("kind") != "ARTIFACT":
            continue
        payload = cap.get("payload") or {}
        path = payload.get("path")
        sealed_sha = payload.get("sha256")
        if not isinstance(path, str) or not isinstance(
                sealed_sha, str):
            continue  # post-hoc fold artifact (sha256=None) —
                       # nothing to verify
        if base_dir and not os.path.isabs(path):
            full_path = os.path.join(base_dir, path)
        else:
            full_path = path
        # Even an absolute path may want to be reinterpreted under
        # ``base_dir`` when the report was relocated. Try both.
        if not os.path.exists(full_path) and base_dir:
            alt = os.path.join(base_dir, os.path.basename(path))
            if os.path.exists(alt):
                full_path = alt
        if not os.path.exists(full_path):
            missing.append({
                "path": path, "full_path": full_path,
                "sealed_sha256": sealed_sha,
            })
            continue
        checked += 1
        actual = _sha256_path(full_path)
        if actual == sealed_sha:
            ok += 1
        else:
            mismatch.append({
                "path": path,
                "sealed_sha256": sealed_sha,
                "on_disk_sha256": actual,
            })
    if missing:
        verdict = "PARTIAL"
    elif mismatch:
        verdict = "BAD"
    elif checked == 0:
        verdict = "EMPTY"
    else:
        verdict = "OK"
    return {
        "verdict": verdict,
        "checked": checked,
        "ok": ok,
        "mismatch": mismatch,
        "missing": missing,
    }


def verify_meta_manifest_on_disk(manifest: dict[str, Any],
                                  *, base_dir: str,
                                  ) -> dict[str, Any]:
    """Verify the meta-artifacts named in a META_MANIFEST view
    against their on-disk bytes.

    ``manifest`` is the parsed contents of ``meta_manifest.json``
    (the secondary view). Each capsule of kind ``META_MANIFEST``
    in the manifest's ``capsules`` list carries a payload with
    ``meta_artifacts: [{path, sha256, n_bytes}, ...]``; this
    function re-reads each path under ``base_dir`` and re-hashes.

    Returns the same shape as ``verify_artifacts_on_disk``.
    """
    checked = 0
    ok = 0
    mismatch: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for cap in manifest.get("capsules", []):
        if cap.get("kind") != "META_MANIFEST":
            continue
        payload = cap.get("payload") or {}
        for entry in (payload.get("meta_artifacts") or []):
            path = entry.get("path")
            sealed_sha = entry.get("sha256")
            if not isinstance(path, str) or not isinstance(
                    sealed_sha, str):
                continue
            full_path = os.path.join(base_dir, path) if (
                not os.path.isabs(path)) else path
            if not os.path.exists(full_path):
                missing.append({
                    "path": path,
                    "full_path": full_path,
                    "sealed_sha256": sealed_sha,
                })
                continue
            checked += 1
            actual = _sha256_path(full_path)
            if actual == sealed_sha:
                ok += 1
            else:
                mismatch.append({
                    "path": path,
                    "sealed_sha256": sealed_sha,
                    "on_disk_sha256": actual,
                })
    if missing:
        verdict = "PARTIAL"
    elif mismatch:
        verdict = "BAD"
    elif checked == 0:
        verdict = "EMPTY"
    else:
        verdict = "OK"
    return {
        "verdict": verdict,
        "checked": checked,
        "ok": ok,
        "mismatch": mismatch,
        "missing": missing,
    }


__all__ += [
    "verify_artifacts_on_disk",
    "verify_meta_manifest_on_disk",
]
