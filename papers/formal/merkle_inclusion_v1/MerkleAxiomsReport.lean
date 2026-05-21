-- W87 / P3 #48 — surfaces the exact axioms each theorem depends on.
-- Run via: `lake env lean MerkleAxiomsReport.lean`
-- Expected output: every theorem depends on
--   propext, Classical.choice, Quot.sound (Lean core),
--   Merkle.Hash, Merkle.Hash.inhabitedInst, Merkle.hashPair  (declaration axioms)
-- And `inclusion_complete` / `merkle_inclusion_iff` additionally depend on
--   Merkle.hashPair_injective  (the only PROPERTY axiom).
-- `inclusion_sound` must NOT depend on `Merkle.hashPair_injective`.

import MerkleInclusion

#print axioms Merkle.inclusion_sound
#print axioms Merkle.inclusion_complete
#print axioms Merkle.merkle_inclusion_iff
