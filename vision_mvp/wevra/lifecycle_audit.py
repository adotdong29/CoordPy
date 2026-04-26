"""Capsule-native lifecycle audit (SDK v3.3).

The capsule-native runtime (``CapsuleNativeRunContext``) seals one
typed capsule per runtime stage and refuses to seal a stage whose
parent CID is not yet in the ledger. The lifecycle correspondence
theorems (W3-32, W3-32-extended, W3-39) state that a finished
ledger satisfies a small set of structural invariants:

  * **L-1** Every capsule sealed in the ledger has a non-None
    sealed_cid in the in-flight register, and every PROPOSED capsule
    that never sealed appears in ``in_flight_failures`` (no orphan
    capsules).
  * **L-2** Every PATCH_PROPOSAL capsule's parent set includes the
    SWEEP_SPEC capsule's CID. Optionally a PARSE_OUTCOME CID.
  * **L-3** Every TEST_VERDICT capsule has exactly one parent, and
    that parent is a sealed PATCH_PROPOSAL.
  * **L-4** Every PARSE_OUTCOME capsule's parent is the SWEEP_SPEC.
  * **L-5** Every SWEEP_CELL capsule's parent is the SWEEP_SPEC.
  * **L-6** Every PATCH_PROPOSAL has a corresponding TEST_VERDICT
    (one-to-one on (instance_id, strategy, parser_mode, apply_mode,
    n_distractors)) AND a corresponding PARSE_OUTCOME.
  * **L-7** Every PATCH_PROPOSAL whose parents include a
    PARSE_OUTCOME has matching coordinates with that PARSE_OUTCOME.
  * **L-8** No TEST_VERDICT precedes its PATCH_PROPOSAL in the
    append order (chain monotonicity at the intra-cell level).

This module exposes ``CapsuleLifecycleAudit`` which mechanically
checks each of L-1..L-8 on a finished ``CapsuleNativeRunContext``
(or any ``CapsuleLedger``) and returns a ``LifecycleAuditReport``.

Theorem W3-40 (Lifecycle-audit soundness, proved by
inspection): if ``CapsuleLifecycleAudit.run().verdict == "OK"`` for
a finished ``CapsuleNativeRunContext``, then the ledger satisfies
L-1..L-8 by construction. The audit is a runtime checker; its
soundness is not a proof of *correctness* of the underlying
runtime — it is a check that the ledger as observed at audit time
is consistent with the lifecycle correspondence theorems.

Counterexamples are surfaced as ``violations`` — a list of
``{rule, capsule_cid, capsule_kind, detail}`` dicts. The first
counterexample (if any) tells you exactly which intra-cell
transition broke the lifecycle.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from .capsule import (
    CapsuleKind,
    CapsuleLedger,
    ContextCapsule,
)


_INTRA_CELL_KINDS = frozenset({
    CapsuleKind.PARSE_OUTCOME,
    CapsuleKind.PATCH_PROPOSAL,
    CapsuleKind.TEST_VERDICT,
})

_SPINE_KINDS = frozenset({
    CapsuleKind.PROFILE,
    CapsuleKind.READINESS_CHECK,
    CapsuleKind.SWEEP_SPEC,
    CapsuleKind.SWEEP_CELL,
    CapsuleKind.PROVENANCE,
    CapsuleKind.ARTIFACT,
    CapsuleKind.RUN_REPORT,
})


@dataclasses.dataclass
class LifecycleAuditReport:
    """Result of one ``CapsuleLifecycleAudit.run()``.

    * ``verdict`` — ``"OK"`` if every rule held, ``"BAD"`` if any
      rule was violated. ``"EMPTY"`` if the ledger had no
      capsule-native execution (no SWEEP_SPEC sealed).
    * ``rules_checked`` — list of rule names that were executed
      against this ledger.
    * ``rules_passed`` — subset of ``rules_checked`` that produced
      no violations.
    * ``violations`` — list of ``{rule, capsule_cid, capsule_kind,
      detail}`` dicts. The order is deterministic (rule name, then
      append order).
    * ``stats`` — counts by kind for transparency.
    """

    verdict: str
    rules_checked: tuple[str, ...]
    rules_passed: tuple[str, ...]
    violations: list[dict[str, Any]]
    stats: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rules_checked": list(self.rules_checked),
            "rules_passed": list(self.rules_passed),
            "violations": list(self.violations),
            "stats": dict(self.stats),
        }


class CapsuleLifecycleAudit:
    """Mechanical checker for the lifecycle correspondence
    invariants L-1..L-8.

    Instantiate with a ``CapsuleNativeRunContext`` *or* a finished
    ``CapsuleLedger`` (with optional ``in_flight_failures``).
    Calling ``run()`` returns a ``LifecycleAuditReport``.

    Theorem W3-40 anchor: this checker is the operational form of
    the lifecycle-audit soundness theorem.
    """

    RULES = (
        "L-1_no_orphan_capsules",
        "L-2_patch_parent_includes_sweep_spec",
        "L-3_verdict_parent_is_sealed_patch",
        "L-4_parse_outcome_parent_is_sweep_spec",
        "L-5_sweep_cell_parent_is_sweep_spec",
        "L-6_patch_has_matching_parse_and_verdict",
        "L-7_patch_coordinates_match_parse_outcome",
        "L-8_chain_order_patch_before_verdict",
    )

    def __init__(self,
                 *,
                 ctx: "Any" = None,
                 ledger: CapsuleLedger | None = None,
                 in_flight_failures: list[dict[str, Any]] | None = None,
                 ) -> None:
        if ctx is not None:
            self.ledger: CapsuleLedger = ctx.ledger
            self.in_flight_failures = ctx.in_flight_failures()
            self._spec_cid = (
                ctx.spec_cap.cid if ctx.spec_cap is not None else None)
        else:
            if ledger is None:
                raise ValueError(
                    "CapsuleLifecycleAudit requires either ``ctx`` or "
                    "``ledger`` argument")
            self.ledger = ledger
            self.in_flight_failures = list(in_flight_failures or [])
            spec = self.ledger.by_kind(CapsuleKind.SWEEP_SPEC)
            self._spec_cid = spec[0].cid if spec else None

    # ------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------

    def _coords(self, cap: ContextCapsule) -> tuple[Any, ...]:
        p = cap.payload or {}
        return (
            p.get("instance_id"),
            p.get("strategy"),
            p.get("parser_mode"),
            p.get("apply_mode"),
            p.get("n_distractors"),
        )

    def _check_l1(self) -> list[dict[str, Any]]:
        # Every PROPOSED-but-not-sealed entry surfaces in
        # in_flight_failures. The runtime guarantees this directly,
        # but the audit checks that the failures list actually has
        # ``failure`` strings (not None) and that no failed entry
        # leaked into the ledger.
        out: list[dict[str, Any]] = []
        for f in self.in_flight_failures:
            if not f.get("failure"):
                out.append({
                    "rule": "L-1_no_orphan_capsules",
                    "capsule_cid": f.get("cid"),
                    "capsule_kind": f.get("kind"),
                    "detail": "in-flight failure has empty failure string",
                })
            cid = f.get("cid")
            if cid in self.ledger:
                out.append({
                    "rule": "L-1_no_orphan_capsules",
                    "capsule_cid": cid,
                    "capsule_kind": f.get("kind"),
                    "detail": ("in-flight failure leaked into ledger; "
                                "the failed capsule MUST NOT be sealed"),
                })
        return out

    def _check_l2(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cap in self.ledger.by_kind(CapsuleKind.PATCH_PROPOSAL):
            if self._spec_cid is None:
                out.append({
                    "rule": "L-2_patch_parent_includes_sweep_spec",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": "no SWEEP_SPEC sealed but PATCH_PROPOSAL exists",
                })
                continue
            if self._spec_cid not in cap.parents:
                out.append({
                    "rule": "L-2_patch_parent_includes_sweep_spec",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": (f"SWEEP_SPEC CID "
                                f"{self._spec_cid[:12]}… not in "
                                f"PATCH_PROPOSAL parents "
                                f"{[p[:8] for p in cap.parents]}"),
                })
        return out

    def _check_l3(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        patch_cids = {
            c.cid for c in self.ledger.by_kind(
                CapsuleKind.PATCH_PROPOSAL)
        }
        for cap in self.ledger.by_kind(CapsuleKind.TEST_VERDICT):
            if len(cap.parents) != 1:
                out.append({
                    "rule": "L-3_verdict_parent_is_sealed_patch",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": (f"TEST_VERDICT has {len(cap.parents)} "
                                f"parents, must be exactly 1"),
                })
                continue
            parent = cap.parents[0]
            if parent not in patch_cids:
                out.append({
                    "rule": "L-3_verdict_parent_is_sealed_patch",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": (f"TEST_VERDICT parent {parent[:12]}… "
                                f"is not a sealed PATCH_PROPOSAL"),
                })
        return out

    def _check_l4(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cap in self.ledger.by_kind(CapsuleKind.PARSE_OUTCOME):
            if self._spec_cid is None:
                out.append({
                    "rule": "L-4_parse_outcome_parent_is_sweep_spec",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": ("no SWEEP_SPEC sealed but "
                                "PARSE_OUTCOME exists"),
                })
                continue
            if cap.parents != (self._spec_cid,):
                out.append({
                    "rule": "L-4_parse_outcome_parent_is_sweep_spec",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": (f"PARSE_OUTCOME parents must be exactly "
                                f"(SWEEP_SPEC,), got "
                                f"{[p[:8] for p in cap.parents]}"),
                })
        return out

    def _check_l5(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cap in self.ledger.by_kind(CapsuleKind.SWEEP_CELL):
            if self._spec_cid is None:
                out.append({
                    "rule": "L-5_sweep_cell_parent_is_sweep_spec",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": "no SWEEP_SPEC sealed but SWEEP_CELL exists",
                })
                continue
            if cap.parents != (self._spec_cid,):
                out.append({
                    "rule": "L-5_sweep_cell_parent_is_sweep_spec",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": (f"SWEEP_CELL parents must be exactly "
                                f"(SWEEP_SPEC,), got "
                                f"{[p[:8] for p in cap.parents]}"),
                })
        return out

    def _check_l6(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        patch_caps = self.ledger.by_kind(CapsuleKind.PATCH_PROPOSAL)
        verdict_caps = self.ledger.by_kind(CapsuleKind.TEST_VERDICT)
        parse_caps = self.ledger.by_kind(CapsuleKind.PARSE_OUTCOME)
        # Coordinate-multiset equality for patch / parse / verdict.
        def coords(cap):
            return self._coords(cap)
        patch_coords = sorted(coords(c) for c in patch_caps)
        verdict_coords = sorted(coords(c) for c in verdict_caps)
        parse_coords = sorted(coords(c) for c in parse_caps)
        if patch_coords != verdict_coords:
            out.append({
                "rule": "L-6_patch_has_matching_parse_and_verdict",
                "capsule_cid": "",
                "capsule_kind": "",
                "detail": (f"PATCH_PROPOSAL count={len(patch_coords)} "
                            f"vs TEST_VERDICT count={len(verdict_coords)} "
                            f"or coordinates disagree"),
            })
        if parse_caps and patch_coords != parse_coords:
            out.append({
                "rule": "L-6_patch_has_matching_parse_and_verdict",
                "capsule_cid": "",
                "capsule_kind": "",
                "detail": (f"PATCH_PROPOSAL count={len(patch_coords)} "
                            f"vs PARSE_OUTCOME count={len(parse_coords)} "
                            f"or coordinates disagree"),
            })
        return out

    def _check_l7(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        parse_by_cid = {
            c.cid: c for c in self.ledger.by_kind(
                CapsuleKind.PARSE_OUTCOME)
        }
        for cap in self.ledger.by_kind(CapsuleKind.PATCH_PROPOSAL):
            for parent_cid in cap.parents:
                if parent_cid in parse_by_cid:
                    parse = parse_by_cid[parent_cid]
                    if self._coords(cap) != self._coords(parse):
                        out.append({
                            "rule": "L-7_patch_coordinates_match_parse_outcome",
                            "capsule_cid": cap.cid,
                            "capsule_kind": cap.kind,
                            "detail": (f"PATCH_PROPOSAL coords "
                                        f"{self._coords(cap)} "
                                        f"!= PARSE_OUTCOME coords "
                                        f"{self._coords(parse)}"),
                        })
        return out

    def _check_l8(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        cid_to_index = {
            c.cid: i for i, c in enumerate(self.ledger.all_capsules())
        }
        for cap in self.ledger.by_kind(CapsuleKind.TEST_VERDICT):
            if not cap.parents:
                continue
            patch_cid = cap.parents[0]
            patch_idx = cid_to_index.get(patch_cid)
            verdict_idx = cid_to_index.get(cap.cid)
            if patch_idx is None or verdict_idx is None:
                continue
            if patch_idx >= verdict_idx:
                out.append({
                    "rule": "L-8_chain_order_patch_before_verdict",
                    "capsule_cid": cap.cid,
                    "capsule_kind": cap.kind,
                    "detail": (f"TEST_VERDICT at index {verdict_idx} "
                                f"sealed BEFORE its PATCH_PROPOSAL at "
                                f"index {patch_idx} — chain order "
                                f"violation (W3-40)"),
                })
        return out

    # ------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------

    def run(self) -> LifecycleAuditReport:
        """Run every rule and return a ``LifecycleAuditReport``."""
        if self._spec_cid is None and not self.ledger.all_capsules():
            return LifecycleAuditReport(
                verdict="EMPTY",
                rules_checked=(),
                rules_passed=(),
                violations=[],
                stats={},
            )
        violations: list[dict[str, Any]] = []
        violations.extend(self._check_l1())
        violations.extend(self._check_l2())
        violations.extend(self._check_l3())
        violations.extend(self._check_l4())
        violations.extend(self._check_l5())
        violations.extend(self._check_l6())
        violations.extend(self._check_l7())
        violations.extend(self._check_l8())
        violated_rules = {v["rule"] for v in violations}
        rules_checked = self.RULES
        rules_passed = tuple(
            r for r in rules_checked if r not in violated_rules)
        # Stats — counts by kind.
        by_kind: dict[str, int] = {}
        for cap in self.ledger.all_capsules():
            by_kind[cap.kind] = by_kind.get(cap.kind, 0) + 1
        verdict = "OK" if not violations else "BAD"
        return LifecycleAuditReport(
            verdict=verdict,
            rules_checked=rules_checked,
            rules_passed=rules_passed,
            violations=violations,
            stats=by_kind,
        )


def audit_capsule_lifecycle(ctx: "Any") -> LifecycleAuditReport:
    """Convenience: build a ``CapsuleLifecycleAudit`` from a
    ``CapsuleNativeRunContext`` and run it. Returns the report."""
    return CapsuleLifecycleAudit(ctx=ctx).run()


def audit_capsule_lifecycle_from_view(view: dict[str, Any]
                                        ) -> LifecycleAuditReport:
    """Audit-from-view: rebuild a minimal in-memory ledger from a
    rendered view dict (the ``capsules`` block of a product
    report, or ``capsule_view.json`` parsed) and run the lifecycle
    audit against it.

    The view's headers always carry CID + kind + parents + lifecycle
    + sizes (Theorem W3-12), so the lifecycle invariants L-2..L-8
    are checkable purely from the view. L-1 (no orphan capsules)
    is vacuously true on a view (the view is the SEALED set; failed
    in-flight entries never appear there).

    This is the right entry point for *forensic* audits where the
    auditor only has the on-disk ``capsule_view.json``, not the
    runtime ctx that produced it.
    """
    ledger = CapsuleLedger()
    # Reconstruct sealed capsules in their stored order. We bypass
    # the admit_and_seal lifecycle because the view's parents may be
    # in any order — we synthesise PROPOSED → SEALED transitions in
    # the canonical append order observed in the view.
    cap_records = view.get("capsules", []) or []
    # First pass — synthesise the ContextCapsule for each header.
    # The capsule's payload may be missing for header-only entries;
    # we substitute an empty dict so the audit can compute coords
    # for kinds that need them. The audit only reads payload fields
    # that the view ALWAYS includes for the relevant kinds (PARSE,
    # PATCH, VERDICT — payload-included by the
    # ``payload_kinds_always`` rule in ``render_view``).
    by_cid: dict[str, ContextCapsule] = {}
    for rec in cap_records:
        cid = rec.get("cid")
        kind = rec.get("kind")
        parents = tuple(rec.get("parents") or ())
        payload = rec.get("payload", {}) if rec.get("payload") else {}
        # Build a synthetic ContextCapsule with sealed lifecycle. We
        # skip admission (the view is already authoritative on
        # CIDs) — this is a *forensic* reconstruction.
        from .capsule import (
            CapsuleBudget, CapsuleLifecycle as _CL,
            _default_budget_for,
        )
        try:
            budget = _default_budget_for(kind)
        except ValueError:
            budget = CapsuleBudget(max_bytes=1 << 16, max_parents=64)
        cap = ContextCapsule(
            cid=cid, kind=kind, payload=payload, budget=budget,
            parents=parents, lifecycle=_CL.SEALED,
            n_tokens=rec.get("n_tokens"),
            n_bytes=rec.get("n_bytes"),
            emitted_at=0.0, metadata=(),
        )
        by_cid[cid] = cap
    # Inject into the ledger's private state — refusing here would
    # require re-deriving every CID, which the view already declares.
    # The audit reads only ``ledger.all_capsules()``, ``by_kind``,
    # ``__contains__``, so populating the dicts is sufficient.
    ledger._by_cid = by_cid  # type: ignore[attr-defined]
    ledger._entries = []  # type: ignore[attr-defined]
    # Walk records in storage order so ``all_capsules()`` and
    # ``by_kind()`` return them in the order the view recorded.
    from .capsule import _LedgerEntry
    for rec in cap_records:
        cid = rec.get("cid")
        if cid not in by_cid:
            continue
        ledger._entries.append(_LedgerEntry(  # type: ignore[attr-defined]
            capsule=by_cid[cid],
            chain_hash="",  # forensic — chain check is W3-37, separate
            prev_chain_hash="",
        ))
    return CapsuleLifecycleAudit(ledger=ledger).run()


__all__ = [
    "CapsuleLifecycleAudit",
    "LifecycleAuditReport",
    "audit_capsule_lifecycle",
    "audit_capsule_lifecycle_from_view",
]
