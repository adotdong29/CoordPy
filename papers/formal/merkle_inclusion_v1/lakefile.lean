-- W87 / P3 #48 — Formal Verification V1
-- Lake build for the Merkle inclusion bidirectional proof.
--
-- Pure Lean 4 stdlib only — no Mathlib dependency.
-- Build with `lake build` from this directory; expected output:
--   Lean 4.13.0 elaborates Merkle.Inclusion and prints no
--   `sorry`/`admit` warnings; `lake env lean MerkleInclusionCheck.lean`
--   prints `#check` outputs for the two main theorems.

import Lake
open Lake DSL

package merkle_inclusion_v1 where
  -- No extra build configuration; defaults are appropriate.

lean_lib MerkleInclusion where
  -- Library root: MerkleInclusion.lean
  roots := #[`MerkleInclusion]

@[default_target]
lean_exe check where
  root := `MerkleInclusionCheck
