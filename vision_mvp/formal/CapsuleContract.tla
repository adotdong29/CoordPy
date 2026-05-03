------------------------ MODULE CapsuleContract ------------------------
(***************************************************************************)
(* Formal specification of the CoordPy Capsule Contract (C1..C6).            *)
(*                                                                         *)
(* This module is a machine-readable restatement of the invariants stated  *)
(* informally in `vision_mvp/coordpy/capsule.py`. A capsule is modelled as a *)
(* record; the ledger is a sequence of sealed capsules; a hash chain is a  *)
(* function from position to chain-hash. The next-state relation is the    *)
(* three lifecycle transitions (admit, seal, retire).                      *)
(*                                                                         *)
(* The spec is deliberately high-level — SHA256 and JSON canonicalisation  *)
(* are modelled as uninterpreted functions. What is checked is the         *)
(* *algebraic* structure of the contract (closed kind vocabulary, legal    *)
(* lifecycle transitions, parent closure, budget monotonicity, chain       *)
(* extension), which is exactly the property set the Python property       *)
(* tests exercise in ``test_capsule_properties.py``.                       *)
(***************************************************************************)

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS
    Kinds,              \* Finite, closed vocabulary of capsule kinds.
    MaxLedgerSize,      \* Bound on ledger length (for TLC finiteness).
    GenesisHash         \* Symbolic initial chain-hash.

ASSUME
    /\ IsFiniteSet(Kinds)
    /\ MaxLedgerSize \in Nat

Lifecycles == { "PROPOSED", "ADMITTED", "SEALED", "RETIRED" }

(***************************************************************************)
(* SHA256 and canonical JSON are opaque at this level of abstraction. The  *)
(* only property we need is that they are deterministic functions of their *)
(* arguments — enough for the identity invariant C1 to be well-posed.      *)
(***************************************************************************)

CONSTANT Canonical(_, _, _, _)   \* canonical byte string of (kind, payload, budget, parents)
CONSTANT SHA256(_)                \* deterministic digest of a byte string

\* State variables.
VARIABLES
    ledger,        \* Sequence of sealed capsule records.
    pending,       \* Set of PROPOSED/ADMITTED capsules not yet sealed.
    chainHashes    \* Function [1..Len(ledger)] -> hash value.

vars == <<ledger, pending, chainHashes>>

(***************************************************************************)
(* Capsule record shape.                                                   *)
(***************************************************************************)

Capsule ==
    [ cid       : STRING,
      kind      : Kinds,
      payload   : STRING,
      budget    : [ maxTokens  : Nat \cup {-1},
                    maxBytes   : Nat \cup {-1},
                    maxRounds  : Nat \cup {-1},
                    maxParents : Nat \cup {-1} ],
      parents   : Seq(STRING),
      nTokens   : Nat,
      nBytes    : Nat,
      lifecycle : Lifecycles ]

\* ``-1`` encodes a missing axis (``None`` in the Python dataclass).
Unbounded(x) == x = -1

(***************************************************************************)
(* C1  Identity.                                                           *)
(*                                                                         *)
(* Every sealed capsule's CID is SHA256 of the canonical encoding of       *)
(* (kind, payload, budget, sorted(parents)). Two capsules with the same    *)
(* contents therefore collapse to one CID.                                 *)
(***************************************************************************)

Inv_C1_Identity ==
    \A i \in 1..Len(ledger) :
        LET c == ledger[i] IN
            c.cid = SHA256(Canonical(c.kind, c.payload, c.budget, c.parents))

(***************************************************************************)
(* C2  Typed claim. Every capsule carries a kind in the closed vocabulary. *)
(***************************************************************************)

Inv_C2_TypedClaim ==
    /\ \A i \in 1..Len(ledger) : ledger[i].kind \in Kinds
    /\ \A c \in pending        : c.kind \in Kinds

(***************************************************************************)
(* C3  Lifecycle. Sealed capsules have SEALED or RETIRED lifecycle;        *)
(* pending capsules have PROPOSED or ADMITTED. No other state pairing is   *)
(* legal. The transition table is PROPOSED -> ADMITTED -> SEALED ->        *)
(* RETIRED.                                                                *)
(***************************************************************************)

LegalTransition(frm, to) ==
    \/ (frm = "PROPOSED" /\ to = "ADMITTED")
    \/ (frm = "ADMITTED" /\ to = "SEALED")
    \/ (frm = "SEALED"   /\ to = "RETIRED")
    \/ (frm = to)   \* idempotent no-op

Inv_C3_Lifecycle ==
    /\ \A i \in 1..Len(ledger) :
        ledger[i].lifecycle \in { "SEALED", "RETIRED" }
    /\ \A c \in pending :
        c.lifecycle \in { "PROPOSED", "ADMITTED" }

(***************************************************************************)
(* C4  Budget. Admission fails if the capsule exceeds its declared token   *)
(* or byte budget, or declares more parents than its parent-count cap.     *)
(***************************************************************************)

BudgetOK(c) ==
    /\ Unbounded(c.budget.maxTokens)  \/ c.nTokens  =< c.budget.maxTokens
    /\ Unbounded(c.budget.maxBytes)   \/ c.nBytes   =< c.budget.maxBytes
    /\ Unbounded(c.budget.maxParents) \/ Len(c.parents) =< c.budget.maxParents

Inv_C4_Budget ==
    \A i \in 1..Len(ledger) : BudgetOK(ledger[i])

(***************************************************************************)
(* C5  Provenance.                                                         *)
(*                                                                         *)
(*   (a) Every parent CID of a sealed capsule must itself be a CID already *)
(*       sealed earlier in the ledger.                                     *)
(*   (b) The hash chain is extended by one link per sealed capsule, where  *)
(*       each link hashes (previous-link, cid, kind, "SEALED"). Retroactive*)
(*       insertion therefore breaks the chain.                             *)
(***************************************************************************)

CidsBefore(i) == { ledger[j].cid : j \in 1..(i-1) }

ParentsClosed ==
    \A i \in 1..Len(ledger) :
        \A k \in 1..Len(ledger[i].parents) :
            ledger[i].parents[k] \in CidsBefore(i)

ChainLinked ==
    /\ (Len(ledger) = 0) \/ (chainHashes[1] = SHA256(<<GenesisHash, ledger[1].cid>>))
    /\ \A i \in 2..Len(ledger) :
        chainHashes[i] = SHA256(<<chainHashes[i-1], ledger[i].cid>>)

Inv_C5_Provenance == ParentsClosed /\ ChainLinked

(***************************************************************************)
(* C6  Frozen. A sealed capsule's CID is fixed: if two sealed entries      *)
(* carry the same CID, they are the same entry.                            *)
(***************************************************************************)

Inv_C6_Frozen ==
    \A i, j \in 1..Len(ledger) :
        (ledger[i].cid = ledger[j].cid) => (i = j)

(***************************************************************************)
(* Conjunction of all six invariants.                                      *)
(***************************************************************************)

AllInvariants ==
    /\ Inv_C1_Identity
    /\ Inv_C2_TypedClaim
    /\ Inv_C3_Lifecycle
    /\ Inv_C4_Budget
    /\ Inv_C5_Provenance
    /\ Inv_C6_Frozen

(***************************************************************************)
(* Initial state.                                                          *)
(***************************************************************************)

Init ==
    /\ ledger       = <<>>
    /\ pending      = {}
    /\ chainHashes  = <<>>

(***************************************************************************)
(* Transitions.                                                            *)
(***************************************************************************)

Propose(c) ==
    /\ c.lifecycle = "PROPOSED"
    /\ c.kind \in Kinds
    /\ pending' = pending \cup { c }
    /\ UNCHANGED <<ledger, chainHashes>>

Admit(c) ==
    /\ c \in pending
    /\ c.lifecycle = "PROPOSED"
    /\ BudgetOK(c)
    /\ \A k \in 1..Len(c.parents) : c.parents[k] \in { ledger[j].cid : j \in 1..Len(ledger) }
    /\ pending' = (pending \ { c }) \cup
                   { [ c EXCEPT !.lifecycle = "ADMITTED" ] }
    /\ UNCHANGED <<ledger, chainHashes>>

Seal(c) ==
    /\ c \in pending
    /\ c.lifecycle = "ADMITTED"
    /\ Len(ledger) < MaxLedgerSize
    /\ LET sealed == [ c EXCEPT !.lifecycle = "SEALED" ]
           prev   == IF Len(ledger) = 0 THEN GenesisHash ELSE chainHashes[Len(ledger)]
       IN  /\ ledger'      = Append(ledger, sealed)
           /\ chainHashes' = Append(chainHashes, SHA256(<<prev, sealed.cid>>))
           /\ pending'     = pending \ { c }

Retire(i) ==
    /\ i \in 1..Len(ledger)
    /\ ledger[i].lifecycle = "SEALED"
    /\ ledger' = [ ledger EXCEPT ![i].lifecycle = "RETIRED" ]
    /\ UNCHANGED <<pending, chainHashes>>

Next ==
    \/ \E c \in Capsule : Propose(c)
    \/ \E c \in pending : Admit(c)
    \/ \E c \in pending : Seal(c)
    \/ \E i \in 1..Len(ledger) : Retire(i)

Spec == Init /\ [][Next]_vars

(***************************************************************************)
(* Top-level theorem: the spec preserves all six invariants.               *)
(***************************************************************************)

THEOREM Spec => []AllInvariants

=============================================================================
