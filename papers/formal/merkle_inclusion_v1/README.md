# Merkle Inclusion V1 — Formal Verification

> **W87 / P3 #48 — Formal Verification of Safety Properties V1.**
> Lean 4 mechanically-checked proof of the bidirectional Merkle
> inclusion property: a leaf is at a directional path iff the
> canonical inclusion proof verifies against the root.

## What this proves

Three theorems in `MerkleInclusion.lean`:

| Theorem | Direction | Crypto assumption used |
|---|---|---|
| `Merkle.inclusion_sound` | membership ⇒ proof verifies | none |
| `Merkle.inclusion_complete` | proof verifies ⇒ membership | `hashPair_injective` |
| `Merkle.merkle_inclusion_iff` | bidirectional | `hashPair_injective` |

The proof uses **no `sorry` and no `admit`**.  The exact axioms
each theorem depends on are surfaced by `MerkleAxiomsReport.lean`:

```
'Merkle.inclusion_sound' depends on axioms:
  [propext, Merkle.Hash, Merkle.hashPair, Quot.sound]
'Merkle.inclusion_complete' depends on axioms:
  [propext, Classical.choice, Merkle.Hash, Merkle.hashPair,
   Merkle.hashPair_injective, Quot.sound]
'Merkle.merkle_inclusion_iff' depends on axioms:
  [propext, Classical.choice, Merkle.Hash, Merkle.hashPair,
   Merkle.hashPair_injective, Quot.sound]
```

`propext`, `Quot.sound`, and `Classical.choice` are Lean 4
foundational axioms (universally accepted; built into the
language).  `Merkle.Hash` and `Merkle.hashPair` are *declaration*
axioms: they assert the existence of an abstract hash type and a
pair-hashing operation, with no commitment to behavior.
`Merkle.hashPair_injective` is the single *property* axiom: the
formal statement of pair-hash collision-resistance.  It is used
by `inclusion_complete` and `merkle_inclusion_iff`, never by
`inclusion_sound`.

## Re-checking the proof from a clean checkout

Prereq: install elan (the official Lean toolchain manager) once
per machine.  This is a single-user install; no sudo:

```bash
curl --proto '=https' --tlsv1.2 -sSf \
  https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
  | sh -s -- -y --default-toolchain leanprover/lean4:v4.13.0
source ~/.elan/env
```

Then, from this directory:

```bash
lake build
```

Expected output (last few lines):

```
✔ [2/6] Built MerkleInclusion
ℹ [3/6] Built MerkleInclusionCheck
info: @inclusion_sound : ∀ {d : Nat} (t : PMerkleTree d) (dir : List Direction) (target : Hash),
  t.memberAt dir target → verifyProof target (t.buildProof dir) = t.root
info: @inclusion_complete : ∀ {d : Nat} (t : PMerkleTree d) (dir : List Direction) (target : Hash),
  dir.length = d → verifyProof target (t.buildProof dir) = t.root → t.memberAt dir target
info: @merkle_inclusion_iff : ∀ {d : Nat} (t : PMerkleTree d) (dir : List Direction),
  dir.length = d → ∀ (target : Hash), t.memberAt dir target ↔ verifyProof target (t.buildProof dir) = t.root
✔ [4/6] Built MerkleInclusionCheck:c.o
✔ [5/6] Built MerkleInclusion:c.o
✔ [6/6] Built check
Build completed successfully.
```

The build also produces an executable that prints a small
self-report:

```bash
./.lake/build/bin/check
# W87 / P3 #48 — Merkle inclusion proof:
#   inclusion_sound       : unconditional
#   inclusion_complete    : assumes hashPair_injective
#   merkle_inclusion_iff  : bidirectional
# All theorems elaborated successfully.
```

To re-print the axiom dependencies:

```bash
lake env lean MerkleAxiomsReport.lean
```

## Toolchain

Pinned to `leanprover/lean4:v4.13.0` in `lean-toolchain`.  Pure
Lean 4 stdlib — no Mathlib.

## Files

| File | Purpose |
|---|---|
| `lean-toolchain` | Toolchain pin |
| `lakefile.lean` | Lake build manifest |
| `MerkleInclusion.lean` | Proof source |
| `MerkleInclusionCheck.lean` | Default `lake build` target (runs `#check`) |
| `MerkleAxiomsReport.lean` | Surfaces the per-theorem axiom dependencies |
| `Coupling.md` | Code ↔ formal coupling: maps the Lean spec to the Python `MerkleHashTreeV1` |
| `README.md` | This file |

## Honesty bound

* Binary trees (fanout 2).  The Python `MerkleHashTreeV1` uses a
  configurable fanout (default 4); the *property* is identical
  for any fanout but the binary case is the canonical formal
  model.  See `Coupling.md` for the explicit fanout-2 mapping.
* The Lean `hashPair` is abstract; the Python `_sha256_hex` is
  concrete.  The coupling asserts that SHA-256 satisfies the
  axiom `hashPair_injective` in practice (collision-resistance
  of SHA-256 is a standard cryptographic assumption).
* The `Coupling.md` document lists every implementation choice
  the Python code makes that the Lean proof does not cover (e.g.,
  the canonical-JSON serialisation, the node-payload schema
  `{"kind": "w82_merkle_node_v1", "children": [...]}`).
