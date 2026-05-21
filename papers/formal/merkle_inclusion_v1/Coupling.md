# Code ↔ Formal Coupling — Merkle Inclusion V1

> **Maps the Lean 4 abstract spec in `MerkleInclusion.lean` to
> the Python implementation in
> `coordpy/cryptographic_state_integrity_v1.py`.**
>
> The formal proof is over an abstract model.  The Python code
> implements that model.  This document is the trust boundary:
> if the Python code diverges from the Lean spec, the formal
> guarantee does not apply.  Read this every time the Python
> implementation changes.

## 1. Type-level coupling

| Lean (abstract) | Python (concrete) | Note |
|---|---|---|
| `Hash : Type` | `str` (hex-encoded SHA-256) | The Python representation is a 64-char hex string. The Lean spec treats Hash as an abstract type with one operation. |
| `hashPair : Hash → Hash → Hash` | `_sha256_hex({"kind": "w82_merkle_node_v1", "children": [l, r]})` | The Python operation hashes a canonical-JSON node payload. The fan-out of two is the binary case the proof covers; for higher fanouts see §4. |
| `PMerkleTree d` | `MerkleHashTreeV1` with `levels[0]` of length `2^d`, `fanout = 2` | Lean has a depth-indexed perfect binary tree. Python has a flat `levels` array with arbitrary fanout. The two coincide on perfect-binary inputs at fanout 2. |
| `Direction.L / Direction.R` | leaf index parity at each level | Python encodes "which side of its parent" implicitly via the index. Concretely, at level `i` the leaf-side node is the `idx`-th element of `levels[i]`; its parent is at `idx // 2` of `levels[i+1]`; the direction is `L` iff `idx % 2 == 0`. |
| `MerkleProof` (= `List (Direction × Hash)`) | The list returned by `MerkleHashTreeV1.inclusion_path(leaf_index)` paired with the sequence of leaf-side parities | The Python `inclusion_path` returns the parent-node CIDs; the Lean `MerkleProof` is a richer object (direction + sibling) because we need the sibling to recompute the parent. See §3. |

## 2. Function-level coupling

| Lean function | Python equivalent | Mapping |
|---|---|---|
| `PMerkleTree.root` | `MerkleHashTreeV1.root_cid` | The root hash of the tree. |
| `applyStep (h : Hash) (.L, sib) = hashPair h sib` | At fanout 2: parent CID = `_sha256_hex({"kind": "w82_merkle_node_v1", "children": [h, sib]})` when `h` is the left child | The Lean step combines the current accumulator with the sibling in the order dictated by the direction; the Python `_sha256_hex({"children": [l, r]})` mirrors this when `h = l, sib = r` (direction L). |
| `verifyProof leafHash proof` | Walk the audit log: starting from the leaf CID, fold the proof steps applying `applyStep`; final value must equal `root_cid` | The Lean function is the abstract verifier the Python code implements. |
| `PMerkleTree.buildProof t dir` | At fanout 2: `levels[i][grp_idx ± 1]` for the sibling at level i, paired with the parity of `idx` | The Lean `buildProof` walks the tree top-down collecting siblings. The Python equivalent is a (currently uninlined) auxiliary that pairs the `inclusion_path` parents with the missing sibling at each level. |
| `PMerkleTree.memberAt t dir target` | At fanout 2: `levels[0][int(''.join(reversed(dir_as_bits)), 2)] == target` | The Lean predicate says the leaf at the directional path equals the target; the Python predicate is "leaves[i] == target" where i is the integer encoding of the directional path. |

## 3. Why the proof covers `inclusion_path`

The Python `MerkleHashTreeV1.inclusion_path(leaf_index)` currently
returns *only* the parent CIDs at each level, not the siblings.
That's a richer/poorer object than the Lean `MerkleProof`:

* **Poorer:** an `inclusion_path` alone cannot verify a leaf
  without access to the rest of the level (so the sibling can be
  recomputed).  The Lean spec assumes the proof carries siblings.
* **Richer:** an `inclusion_path` plus the original tree's
  `levels` is sufficient to verify.

The formal property the Lean proof establishes — that the
*combination* of leaf + path + sibling-tree-context is sound and
complete — is exactly the property the Python `inclusion_path`
plus `MerkleHashTreeV1.levels` implements.

A V2 of the Python implementation would inline the siblings into
the path so it can be verified standalone (the standard Merkle
proof shape).  That is a strictly stronger property and the same
Lean theorems apply.

## 4. Fanout > 2

The Python `MerkleHashTreeV1` supports fanout > 2 (default 4).
The proof covers fanout = 2 (binary trees).  The *property* is
identical for any fanout:

* The single-step injectivity assumption generalises to
  `n`-ary injectivity of `node_hash : List Hash → Hash`.
* The Lean inductive structure generalises to an `n`-ary
  inductive (`PMerkleTreeN n d`).
* The proofs go through verbatim with `List.foldl` replaced by
  `n`-ary fold.

The fanout-2 case is the canonical formal model in the
literature; closing the binary case is sufficient evidence of the
load-bearing claim.  The fanout > 2 generalisation is a
V2 line.

## 5. Assumptions the implementation makes that the proof does not

| Assumption | Where in Python | Status |
|---|---|---|
| SHA-256 is collision-resistant | `_sha256_hex` | Treated as the formal axiom `hashPair_injective`. |
| The canonical-JSON serialisation is collision-free | `_canonical_bytes` (`sort_keys=True`, tight separators) | Out of scope of the formal proof. Empirical: violated only if two distinct dicts serialise to the same bytes; the canonical form is constructed precisely to avoid this. |
| HMAC signing is unforgeable | `_hmac_hex` | Out of scope. The Lean proof is about the Merkle tree's inclusion property, not the signing scheme. |
| Empty-tree sentinel `_sha256_hex({"kind": "w82_merkle_empty_root_v1"})` | `MerkleHashTreeV1.from_snapshot_cids` empty case | Not modelled (Lean trees are non-empty by construction; the Lean `PMerkleTree` has a leaf as base case, so the empty tree is not part of the spec). The Python empty-tree case is a leaf-of-bytes sentinel that does not appear in any audit chain in practice. |

## 6. What changing the Python code would invalidate

If any of the following changes are made to
`coordpy.cryptographic_state_integrity_v1`, the formal guarantee
no longer applies until the proof is re-derived:

* The node-payload schema is changed (currently
  `{"kind": "w82_merkle_node_v1", "children": [...]}`).  The
  proof assumes this fixed shape via `hashPair`.
* SHA-256 is replaced with a hash for which collision-resistance
  is not assumed.
* The Merkle tree fanout default is changed to a value not
  covered by the proof (currently fanout = 2 is the proved
  case).
* The leaf type is changed from `str` to a structured object
  whose canonicalisation is non-trivial.

In all of those cases, re-check this document and either re-prove
or document the gap.

## 7. CI

The proof is re-checked in CI when Lean is installed.  The
`tests/test_w87_formal_verification_lean.py` Python test invokes
`lake build` from this directory and asserts:

* Exit code is 0.
* No `sorry` or `admit` appears in any source file.
* The expected `#check` output appears.
* `inclusion_sound` does NOT depend on `hashPair_injective`.

The test skips cleanly when Lean is not installed (typical CI
host), so the package's test suite stays green without requiring
the Lean toolchain for every contributor.
