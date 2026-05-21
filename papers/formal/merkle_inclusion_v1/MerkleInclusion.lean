/-
W87 / P3 #48 — Formal Verification V1
Merkle Inclusion: bidirectional soundness + completeness.

Depth-indexed perfect binary Merkle tree.  The hash is abstract;
the only crypto assumption used is collision-resistance of the
pair hash (axiom `hashPair_injective`).

Crypto declarations (surfaced as Lean axioms so the proof never
commits to a specific hash realization)

  * `Hash : Type`                       declaration axiom.
  * `Hash.inhabitedInst`                declaration axiom (instance).
  * `hashPair : Hash → Hash → Hash`     declaration axiom.
  * `hashPair_injective : ...`          property axiom (only one
                                        used in the proof body).

These axioms are jointly consistent: one concrete model is
`Hash := Nat`, `hashPair := Nat.pair`, which is injective.

Load-bearing theorems

  * `inclusion_sound`     unconditional w.r.t. property axioms.
  * `inclusion_complete`  uses `hashPair_injective`.
  * `merkle_inclusion_iff` the bidirectional statement.

Honest scope (W87)

  * Binary trees (fanout 2).  The Python `MerkleHashTreeV1` uses
    a configurable fanout (default 4); the *property* is identical
    for any fanout but the binary case is the canonical formal
    model.  See `Coupling.md`.
-/

namespace Merkle

/-! ## Abstract hash and direction -/

axiom Hash : Type
axiom Hash.inhabitedInst : Inhabited Hash
attribute [instance] Hash.inhabitedInst
axiom hashPair : Hash → Hash → Hash

/-- Direction at a binary node: left or right child. -/
inductive Direction
  | L
  | R
  deriving DecidableEq, Repr

/-! ## Depth-indexed perfect binary Merkle tree -/

/-- A perfect binary tree whose depth is encoded in the type. -/
inductive PMerkleTree : Nat → Type
  | leaf : Hash → PMerkleTree 0
  | node {d : Nat} : PMerkleTree d → PMerkleTree d → PMerkleTree (d + 1)

/-! ## Proof, applyStep, verify -/

abbrev ProofStep := Direction × Hash
abbrev MerkleProof := List ProofStep

/-- The root hash. -/
noncomputable def PMerkleTree.root : {d : Nat} → PMerkleTree d → Hash
  | 0, .leaf h => h
  | _ + 1, .node l r => hashPair l.root r.root

/-- Apply one proof step: combine the current hash with the
    sibling on the correct side. -/
noncomputable def applyStep (h : Hash) : ProofStep → Hash
  | (.L, sib) => hashPair h sib
  | (.R, sib) => hashPair sib h

/-- Verify a proof: fold the steps from leaf upward. -/
noncomputable def verifyProof (leafHash : Hash) (proof : MerkleProof) : Hash :=
  proof.foldl applyStep leafHash

/-- `t.memberAt dir target` says following `dir` from the root of
    `t` lands at a leaf whose hash equals `target`.  When `dir`
    has the wrong length, the result is `False`. -/
def PMerkleTree.memberAt : {d : Nat} → PMerkleTree d → List Direction → Hash → Prop
  | 0, .leaf h, [], target => h = target
  | 0, .leaf _, _ :: _, _ => False
  | _ + 1, .node _ _, [], _ => False
  | _ + 1, .node l _, .L :: rest, target => l.memberAt rest target
  | _ + 1, .node _ r, .R :: rest, target => r.memberAt rest target

/-- Canonical inclusion-proof builder.  Walks `dir` from root to
    leaf, collecting the sibling root at each step.  Returns the
    proof leaf-side-first to match `verifyProof`. -/
noncomputable def PMerkleTree.buildProof :
    {d : Nat} → PMerkleTree d → List Direction → MerkleProof
  | 0, .leaf _, _ => []
  | _ + 1, .node _ _, [] => []
  | _ + 1, .node l r, .L :: rest =>
      l.buildProof rest ++ [(.L, r.root)]
  | _ + 1, .node l r, .R :: rest =>
      r.buildProof rest ++ [(.R, l.root)]

/-! ## Equation lemmas (for simp rewriting) -/

@[simp] theorem root_leaf (h : Hash) : (PMerkleTree.leaf h).root = h := rfl

@[simp] theorem root_node {d : Nat} (l r : PMerkleTree d) :
    (l.node r).root = hashPair l.root r.root := rfl

@[simp] theorem applyStep_L (h sib : Hash) :
    applyStep h (.L, sib) = hashPair h sib := rfl

@[simp] theorem applyStep_R (h sib : Hash) :
    applyStep h (.R, sib) = hashPair sib h := rfl

@[simp] theorem verifyProof_nil (h : Hash) : verifyProof h [] = h := rfl

@[simp] theorem buildProof_leaf (h : Hash) (dir : List Direction) :
    (PMerkleTree.leaf h).buildProof dir = [] := by
  cases dir <;> rfl

@[simp] theorem buildProof_node_nil {d : Nat} (l r : PMerkleTree d) :
    (l.node r).buildProof [] = [] := rfl

@[simp] theorem buildProof_node_L {d : Nat} (l r : PMerkleTree d)
    (rest : List Direction) :
    (l.node r).buildProof (.L :: rest) =
      l.buildProof rest ++ [(.L, r.root)] := rfl

@[simp] theorem buildProof_node_R {d : Nat} (l r : PMerkleTree d)
    (rest : List Direction) :
    (l.node r).buildProof (.R :: rest) =
      r.buildProof rest ++ [(.R, l.root)] := rfl

@[simp] theorem memberAt_leaf (h target : Hash) :
    (PMerkleTree.leaf h).memberAt [] target = (h = target) := rfl

@[simp] theorem memberAt_leaf_cons (h : Hash) (head : Direction)
    (rest : List Direction) (target : Hash) :
    (PMerkleTree.leaf h).memberAt (head :: rest) target = False := by
  cases head <;> rfl

@[simp] theorem memberAt_node_nil {d : Nat} (l r : PMerkleTree d) (target : Hash) :
    (l.node r).memberAt [] target = False := rfl

@[simp] theorem memberAt_node_L {d : Nat} (l r : PMerkleTree d)
    (rest : List Direction) (target : Hash) :
    (l.node r).memberAt (.L :: rest) target = l.memberAt rest target := rfl

@[simp] theorem memberAt_node_R {d : Nat} (l r : PMerkleTree d)
    (rest : List Direction) (target : Hash) :
    (l.node r).memberAt (.R :: rest) target = r.memberAt rest target := rfl

/-! ## Foldl lemma -/

theorem foldl_snoc (init : Hash) (xs : MerkleProof) (y : ProofStep) :
    (xs ++ [y]).foldl applyStep init = applyStep (xs.foldl applyStep init) y := by
  induction xs generalizing init with
  | nil => rfl
  | cons z zs ih => simp [List.foldl_cons, ih]

theorem verifyProof_snoc (init : Hash) (xs : MerkleProof) (y : ProofStep) :
    verifyProof init (xs ++ [y]) = applyStep (verifyProof init xs) y := by
  simp [verifyProof, foldl_snoc]

/-! ## Soundness (unconditional) -/

/-- **Soundness.** The canonically-built proof for any directional
    path of the right length in a tree verifies against that tree's
    root. -/
theorem inclusion_sound {d : Nat} :
    ∀ (t : PMerkleTree d) (dir : List Direction) (target : Hash),
      t.memberAt dir target →
      verifyProof target (t.buildProof dir) = t.root := by
  induction d with
  | zero =>
      intro t dir target h_mem
      cases t with
      | leaf h =>
          cases dir with
          | nil =>
              simp at h_mem
              simp [h_mem]
          | cons _ _ => simp at h_mem
  | succ n ih =>
      intro t dir target h_mem
      cases t with
      | node l r =>
          cases dir with
          | nil => simp at h_mem
          | cons head rest =>
              cases head with
              | L =>
                  simp at h_mem
                  have h_sub := ih l rest target h_mem
                  simp [verifyProof_snoc, h_sub]
              | R =>
                  simp at h_mem
                  have h_sub := ih r rest target h_mem
                  simp [verifyProof_snoc, h_sub]

/-! ## Completeness (collision-resistant) -/

/-- **Collision-resistance axiom.** The hash-pair operation is
    injective on its two arguments.  This is the only PROPERTY
    axiom in the proof — the formal statement of SHA-256 pair-
    hash collision-resistance. -/
axiom hashPair_injective :
    ∀ {a b c d : Hash}, hashPair a b = hashPair c d → a = c ∧ b = d

/-- **Completeness.** If the canonically-built proof for a
    directional path verifies against the root, then the leaf is
    indeed at that path.  Uses `hashPair_injective`. -/
theorem inclusion_complete {d : Nat} :
    ∀ (t : PMerkleTree d) (dir : List Direction) (target : Hash),
      dir.length = d →
      verifyProof target (t.buildProof dir) = t.root →
      t.memberAt dir target := by
  induction d with
  | zero =>
      intro t dir target h_len h_verify
      cases t with
      | leaf h =>
          cases dir with
          | nil =>
              -- h_verify : verifyProof target [] = h, i.e. target = h
              -- goal     : h = target
              simp [verifyProof, PMerkleTree.buildProof,
                    PMerkleTree.root] at h_verify
              exact h_verify.symm
          | cons _ _ => simp at h_len
  | succ n ih =>
      intro t dir target h_len h_verify
      cases t with
      | node l r =>
          cases dir with
          | nil => simp at h_len
          | cons head rest =>
              cases head with
              | L =>
                  simp at h_len
                  -- After simp + verifyProof_snoc + applyStep_L lemmas,
                  -- h_verify : hashPair (verifyProof target (l.buildProof rest)) r.root = hashPair l.root r.root
                  simp [verifyProof_snoc] at h_verify
                  have ⟨h_acc, _⟩ := hashPair_injective h_verify
                  exact ih l rest target h_len h_acc
              | R =>
                  simp at h_len
                  simp [verifyProof_snoc] at h_verify
                  have ⟨_, h_acc⟩ := hashPair_injective h_verify
                  exact ih r rest target h_len h_acc

/-! ## Bidirectional theorem -/

/-- **Bidirectional Merkle inclusion (load-bearing).**

    `t.memberAt dir target ↔ verifyProof target (t.buildProof dir) = t.root`
    for any perfect binary Merkle tree `t` and directional path
    `dir` of the right length.

    Forward (`→`) is `inclusion_sound`, unconditional w.r.t. the
    property axiom.
    Backward (`←`) is `inclusion_complete`, conditional on
    `hashPair_injective`. -/
theorem merkle_inclusion_iff {d : Nat} (t : PMerkleTree d)
    (dir : List Direction) (h_len : dir.length = d) (target : Hash) :
    t.memberAt dir target ↔
    verifyProof target (t.buildProof dir) = t.root := by
  exact ⟨inclusion_sound t dir target,
         inclusion_complete t dir target h_len⟩

end Merkle
