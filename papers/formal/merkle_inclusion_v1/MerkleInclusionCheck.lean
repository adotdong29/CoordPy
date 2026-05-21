-- W87 / P3 #48 — Formal Verification V1 entrypoint.
-- A small driver that prints the elaborated types of the
-- load-bearing theorems.  Lake builds this as the default target
-- so `lake build` exits non-zero if the proof file fails to
-- elaborate.

import MerkleInclusion

open Merkle

#check @inclusion_sound
#check @inclusion_complete
#check @merkle_inclusion_iff

def main : IO Unit := do
  IO.println "W87 / P3 #48 — Merkle inclusion proof:"
  IO.println "  inclusion_sound       : unconditional"
  IO.println "  inclusion_complete    : assumes hashPair_injective"
  IO.println "  merkle_inclusion_iff  : bidirectional"
  IO.println "All theorems elaborated successfully."
