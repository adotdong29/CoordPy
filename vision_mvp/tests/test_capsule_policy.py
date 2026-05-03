"""Capsule admission policy tests (Phase 46).

Locks in the contract on the policy framework that lifts capsule
admission from a fixed budget gate to a **learnable** decision.

The contract:

  * featuriser is deterministic on capsule contents
    (same capsule → same feature dict);
  * `train_admission_policy` is deterministic in `seed`;
  * `BudgetedAdmissionLedger` enforces budget exactly
    (no admit past `budget_tokens`);
  * a learned policy beats `FIFOPolicy` on a synthetic
    train/test split where the hypothesis class is
    realisable (regression-tested);
  * the SDK-v3 default behaviour (FIFO) is preserved when
    no policy is specified.
"""

from __future__ import annotations

import unittest


class CapsulePolicyContractTests(unittest.TestCase):

    def test_featurise_is_deterministic(self):
        from vision_mvp.coordpy import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        from vision_mvp.coordpy.capsule_policy import featurise_capsule

        c1 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"msg": "hi", "n": 1},
            budget=CapsuleBudget(max_tokens=64),
            n_tokens=4,
            metadata={"source_role": "monitor",
                      "claim_kind": "ERROR_RATE_SPIKE"})
        c2 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF,
            payload={"n": 1, "msg": "hi"},  # key order swapped
            budget=CapsuleBudget(max_tokens=64),
            n_tokens=4,
            metadata={"source_role": "monitor",
                      "claim_kind": "ERROR_RATE_SPIKE"})
        f1 = featurise_capsule(c1)
        f2 = featurise_capsule(c2)
        self.assertEqual(f1, f2)
        self.assertEqual(f1["bias"], 1.0)
        self.assertEqual(f1["kind:HANDOFF"], 1.0)
        self.assertEqual(f1["src:monitor"], 1.0)
        self.assertEqual(f1["claim:ERROR_RATE_SPIKE"], 1.0)

    def test_feature_index_is_closed_vocab(self):
        from vision_mvp.coordpy.capsule_policy import (
            feature_index, _KIND_FEATURES,
        )
        idx = feature_index()
        self.assertEqual(idx[0], "bias")
        # Every kind in CapsuleKind.ALL appears as a "kind:K"
        # feature in the index.
        for k in _KIND_FEATURES:
            self.assertIn(f"kind:{k}", idx)

    def test_budgeted_ledger_enforces_token_budget(self):
        from vision_mvp.coordpy import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        from vision_mvp.coordpy.capsule_policy import (
            BudgetedAdmissionLedger, FIFOPolicy, ADMIT, REJECT,
        )

        bal = BudgetedAdmissionLedger(
            budget_tokens=10, policy=FIFOPolicy())
        # Three capsules of 4 tokens each — third should be
        # rejected because 4+4+4 = 12 > 10.
        for i in range(3):
            c = ContextCapsule.new(
                kind=CapsuleKind.HANDOFF,
                payload={"i": i, "msg": "hello"},
                budget=CapsuleBudget(max_tokens=64),
                n_tokens=4,
                metadata={"source_role": "monitor",
                          "claim_kind": "ERROR_RATE_SPIKE"})
            d = bal.offer(c)
            if i < 2:
                self.assertEqual(d, ADMIT, f"capsule {i} expected ADMIT")
            else:
                self.assertEqual(d, REJECT,
                                  f"capsule {i} expected REJECT")
        self.assertEqual(bal.budget_used, 8)
        self.assertEqual(bal.remaining, 2)

    def test_kind_priority_admits_high_priority_only(self):
        from vision_mvp.coordpy import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        from vision_mvp.coordpy.capsule_policy import (
            BudgetedAdmissionLedger, KindPriorityPolicy, ADMIT,
        )

        pol = KindPriorityPolicy(cutoff=2)  # accept top-2 only
        bal = BudgetedAdmissionLedger(
            budget_tokens=1024, policy=pol)
        # DISK_FILL_CRITICAL is rank 0 — admit.
        c1 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"msg": "disk"},
            budget=CapsuleBudget(max_tokens=64), n_tokens=4,
            metadata={"source_role": "sysadmin",
                      "claim_kind": "DISK_FILL_CRITICAL"})
        # SLOW_QUERY is rank > 2 — reject.
        c2 = ContextCapsule.new(
            kind=CapsuleKind.HANDOFF, payload={"msg": "sql"},
            budget=CapsuleBudget(max_tokens=64), n_tokens=4,
            metadata={"source_role": "db_admin",
                      "claim_kind": "SLOW_QUERY_OBSERVED"})
        self.assertEqual(bal.offer(c1), ADMIT)
        self.assertEqual(bal.offer(c2), "REJECT")

    def test_train_admission_policy_is_deterministic_in_seed(self):
        from vision_mvp.coordpy import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        from vision_mvp.coordpy.capsule_policy import (
            train_admission_policy,
        )

        examples = []
        for i in range(20):
            c = ContextCapsule.new(
                kind=CapsuleKind.HANDOFF,
                payload={"i": i},
                budget=CapsuleBudget(max_tokens=64),
                n_tokens=4 + (i % 3),
                metadata={
                    "source_role": (
                        "monitor" if i % 2 == 0 else "network"),
                    "claim_kind": (
                        "ERROR_RATE_SPIKE" if i % 2 == 0
                        else "DISK_FILL_CRITICAL"),
                })
            label = 1 if i % 2 == 0 else 0
            examples.append((c, label))

        p1 = train_admission_policy(
            examples, n_epochs=50, lr=0.5, l2=1e-3, seed=7)
        p2 = train_admission_policy(
            examples, n_epochs=50, lr=0.5, l2=1e-3, seed=7)
        self.assertEqual(p1.weights, p2.weights)
        self.assertEqual(p1.feature_means, p2.feature_means)
        # Different seed shouldn't dramatically change the
        # solution because GD is full-batch + deterministic;
        # but the shuffle order changes, so weights may shift
        # slightly. Just verify the model converges to the
        # *separating* sign on a discriminative feature.
        self.assertGreater(
            p1.weights.get("src:monitor", 0.0), 0.0)

    def test_learned_policy_beats_fifo_on_separable_data(self):
        """Regression test: on a synthetic train/test split where
        the hypothesis class is realisable, the learned policy's
        admit-precision strictly beats FIFO on the test set."""
        from vision_mvp.coordpy import (
            ContextCapsule, CapsuleKind, CapsuleBudget,
        )
        from vision_mvp.coordpy.capsule_policy import (
            BudgetedAdmissionLedger, FIFOPolicy,
            train_admission_policy, ADMIT,
        )

        def _mk_capsule(i: int, causal: bool) -> ContextCapsule:
            return ContextCapsule.new(
                kind=CapsuleKind.HANDOFF,
                payload={"i": i, "c": causal},
                budget=CapsuleBudget(max_tokens=64),
                n_tokens=4,
                metadata={
                    "source_role": (
                        "monitor" if causal else "network"),
                    "claim_kind": (
                        "ERROR_RATE_SPIKE" if causal
                        else "DISK_FILL_CRITICAL"),
                })

        # Train: 60 examples, 50/50 causal/non-causal, perfectly
        # separable on (source_role, claim_kind).
        train = []
        for i in range(60):
            causal = (i % 2 == 0)
            train.append((_mk_capsule(i, causal), int(causal)))
        # Test: another 40 examples, same distribution.
        test_capsules = [_mk_capsule(1000 + i, i % 2 == 0)
                          for i in range(40)]
        test_labels = [int(i % 2 == 0) for i in range(40)]

        learned = train_admission_policy(
            train, n_epochs=200, lr=0.5, l2=1e-3, seed=0)

        def _admit_precision(policy, budget):
            bal = BudgetedAdmissionLedger(
                budget_tokens=budget, policy=policy)
            bal.offer_all_batched(test_capsules)
            admitted = {d.capsule_cid for d in bal.decisions
                         if d.decision == ADMIT}
            adm_idxs = [i for i, c in enumerate(test_capsules)
                         if c.cid in admitted]
            if not adm_idxs:
                return 0.0
            adm_causal = sum(test_labels[i] for i in adm_idxs)
            return adm_causal / len(adm_idxs)

        # At a tight budget, learned should strictly beat FIFO
        # on admit-precision.
        p_learned = _admit_precision(learned, budget=40)
        p_fifo = _admit_precision(FIFOPolicy(), budget=40)
        self.assertGreater(
            p_learned, p_fifo,
            f"learned {p_learned} ≤ FIFO {p_fifo}")
        # And learned should be near-perfect on this realisable
        # task.
        self.assertGreaterEqual(p_learned, 0.95)


if __name__ == "__main__":
    unittest.main()
