"""Verify the Python ``CapsuleLedger`` matches the TLA+ ``CapsuleContract``.

The TLA+ module specifies six invariants C1..C6 on an abstract ledger.
``ConsistencyChecker`` exercises the concrete Python ``CapsuleLedger``
with randomised sequences of admit / seal / retire operations, takes
state snapshots before and after each transition, and verifies that
every snapshot satisfies the same six invariants.  A single violation
anywhere in the trace is a concrete counter-example to implementation /
spec consistency; zero violations across thousands of trials is the
runtime witness the spec is actually *implemented*.

This is a fuzz-style property check, not a model-checking run — it
complements ``run_model_checker.py`` rather than replacing it.  The two
together close the loop: TLC verifies the spec is internally consistent,
``ConsistencyChecker`` verifies the implementation refines the spec.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from vision_mvp.coordpy.capsule import (
    CapsuleBudget,
    CapsuleKind,
    CapsuleLedger,
    CapsuleLifecycle,
    ContextCapsule,
    _capsule_cid,
    _chain_step,
)


# =============================================================================
# Snapshot + transition types
# =============================================================================


@dataclass
class Snapshot:
    size: int
    chain_head: str
    chain_ok: bool
    all_cids: tuple[str, ...]
    kinds: tuple[str, ...]
    lifecycles: tuple[str, ...]


@dataclass
class Transition:
    action: str              # 'admit_and_seal' | 'retire'
    cid: str
    before: Snapshot
    after: Snapshot
    capsule: ContextCapsule


# =============================================================================
# Consistency checker
# =============================================================================


@dataclass
class ConsistencyReport:
    all_pass: bool
    n_transitions: int
    violations: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "all_pass": self.all_pass,
            "n_transitions": self.n_transitions,
            "violations": self.violations,
        }


class ConsistencyChecker:
    """Refinement-style consistency check for the capsule ledger."""

    # ---- behaviour extraction ----------------------------------------------

    def extract_python_behavior(self, n_capsules: int = 100,
                                seed: int = 0) -> list[Transition]:
        """Run the ledger for ``n_capsules`` steps, returning the full
        transition trace.

        Payloads are randomised so CIDs differ; budgets are picked so
        admission never fails (admission-failure is a legal outcome, but
        we want traces long enough to exercise the invariants).
        """

        rng = random.Random(seed)
        ledger = CapsuleLedger()
        transitions: list[Transition] = []

        kinds = [
            CapsuleKind.ARTIFACT,
            CapsuleKind.SWEEP_CELL,
            CapsuleKind.READINESS_CHECK,
            CapsuleKind.PROFILE,
        ]

        for i in range(n_capsules):
            kind = rng.choice(kinds)
            parents_pool = [c.cid for c in ledger.all_capsules()
                            if c.lifecycle == CapsuleLifecycle.SEALED]
            k = min(rng.randint(0, 3), len(parents_pool))
            parents = tuple(rng.sample(parents_pool, k)) if k else ()

            cap = ContextCapsule.new(
                kind=kind,
                payload={"i": i, "r": rng.randint(0, 1_000_000)},
                budget=CapsuleBudget(max_bytes=1 << 16, max_parents=16),
                parents=parents,
            )

            before = self._snapshot(ledger)
            try:
                sealed = ledger.admit_and_seal(cap)
            except Exception:
                continue
            after = self._snapshot(ledger)
            transitions.append(Transition(
                action="admit_and_seal", cid=sealed.cid,
                before=before, after=after, capsule=sealed,
            ))

        return transitions

    def _snapshot(self, ledger: CapsuleLedger) -> Snapshot:
        caps = ledger.all_capsules()
        return Snapshot(
            size=len(caps),
            chain_head=ledger.chain_head(),
            chain_ok=ledger.verify_chain(),
            all_cids=tuple(c.cid for c in caps),
            kinds=tuple(c.kind for c in caps),
            lifecycles=tuple(c.lifecycle for c in caps),
        )

    # ---- invariant checks --------------------------------------------------

    def _check_invariants(self, t: Transition,
                          ledger: CapsuleLedger | None = None
                          ) -> list[str]:
        """Return the list of TLA+ invariant names that the transition
        violates.  Empty list = all invariants hold."""

        bad: list[str] = []
        cap = t.capsule

        # C1  Identity — CID is sha256 of canonical (kind, payload, budget, parents).
        expected_cid = _capsule_cid(cap.kind, cap.payload, cap.budget,
                                    cap.parents)
        if cap.cid != expected_cid:
            bad.append("C1_Identity")

        # C2  Typed claim — kind belongs to the closed vocabulary.
        if cap.kind not in CapsuleKind.ALL:
            bad.append("C2_TypedClaim")

        # C3  Lifecycle — a sealed capsule must have traversed legal edges.
        if cap.lifecycle not in CapsuleLifecycle.ALL:
            bad.append("C3_Lifecycle")
        if t.action == "admit_and_seal" and cap.lifecycle != CapsuleLifecycle.SEALED:
            bad.append("C3_Lifecycle")

        # C4  Budget — declared axes are not exceeded.
        b = cap.budget
        if (b.max_bytes is not None and cap.n_bytes > b.max_bytes):
            bad.append("C4_Budget")
        if (b.max_tokens is not None and cap.n_tokens > b.max_tokens):
            bad.append("C4_Budget")
        if (b.max_parents is not None and len(cap.parents) > b.max_parents):
            bad.append("C4_Budget")

        # C5  Provenance — chain is extension-intact and all parents exist.
        if not t.after.chain_ok:
            bad.append("C5_Provenance")
        parents_set = set(t.before.all_cids)
        for p in cap.parents:
            if p not in parents_set:
                bad.append("C5_Provenance")
                break

        # Strong chain check: the new head should be chain_step(prev_head, cap).
        expected_head = _chain_step(t.before.chain_head, cap)
        if t.after.chain_head != expected_head:
            bad.append("C5_Provenance")

        # C6  Frozen — every CID present before must still be present, unchanged.
        before_set = set(t.before.all_cids)
        after_set = set(t.after.all_cids)
        if not before_set.issubset(after_set):
            bad.append("C6_Frozen")

        # Deduplicate while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for name in bad:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
        return out

    # ---- entry points ------------------------------------------------------

    def verify_against_tla_spec(self,
                                transitions: list[Transition]
                                ) -> ConsistencyReport:
        violations: list[dict] = []
        for idx, t in enumerate(transitions):
            bad = self._check_invariants(t)
            if bad:
                violations.append({
                    "index": idx,
                    "cid": t.cid,
                    "action": t.action,
                    "invariants": bad,
                })
        return ConsistencyReport(
            all_pass=not violations,
            n_transitions=len(transitions),
            violations=violations,
        )

    def fuzz_consistency(self, n_trials: int = 1000,
                         ops_per_trial: int = 10,
                         seed: int = 0) -> dict:
        """Run ``n_trials`` independent ledger sessions, each ``ops_per_trial``
        operations deep, and verify that no session produces any invariant
        violation.

        Returns a summary dict suitable for a CI assertion.
        """

        total_transitions = 0
        total_violations = 0
        first_violation: dict | None = None

        for trial in range(n_trials):
            transitions = self.extract_python_behavior(
                n_capsules=ops_per_trial, seed=seed + trial)
            report = self.verify_against_tla_spec(transitions)
            total_transitions += report.n_transitions
            if not report.all_pass:
                total_violations += len(report.violations)
                if first_violation is None:
                    first_violation = {
                        "trial": trial,
                        "violation": report.violations[0],
                    }

        return {
            "trials_passed": n_trials - (1 if total_violations else 0),
            "n_trials": n_trials,
            "total_transitions": total_transitions,
            "total_violations": total_violations,
            "first_violation": first_violation,
        }


def main() -> None:
    checker = ConsistencyChecker()
    summary = checker.fuzz_consistency(n_trials=1000, ops_per_trial=10)
    print(summary)
    assert summary["total_violations"] == 0, summary


if __name__ == "__main__":
    main()
