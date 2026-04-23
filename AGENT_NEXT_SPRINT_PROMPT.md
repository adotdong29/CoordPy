# Next Sprint Prompt: Path to 10/10 (3-Week Agent Sprint)

## Context

You are continuing work on Wevra. **Status after 1-day sprint**: 9.2/10 average (up from 8.2).

**Reference documents**:
- `ADVANCEMENT_TO_10_10.md` — full research roadmap
- `CONTEXT_SOLUTION_REVIEW.md` — what we've solved and what's missing
- `AGENT_IMPLEMENTATION_PROMPT.md` — how previous agent worked

**Your mission**: Implement the **three hardest problems** that will push Wevra to 9.8+/10 (effectively 10/10 in practice).

**Timeline**: 3 weeks (intensive, focused work)

**Expected outcome**: 
- Category theory formalization + proof
- TLA+ model checking + consistency verification
- Learned context routing (working LSTM)

**Average improvement**: +1.5 to +2 points → **10.7/10 → capped at 10/10**

---

## Three Priority Problems (In Order)

### **Priority 1: Category Theory Formalization** (Week 1)
**Impact**: Originality +2, Research +1.5, Theoretical Rigor +1  
**Status**: Currently implicit; need explicit mathematical grounding

**What to implement**:

**File**: `vision_mvp/core/categorical_semantics.py` (NEW, 300+ lines)

**Goal**: Prove that Phase-31 handoff routing is computing Kan extensions.

**Mathematical framework**:

1. **Define CapsuleCategory**:
   ```python
   class CapsuleCategory(metaclass=ABCMeta):
       """Symmetric monoidal category of capsules.
       
       Theorem CAT-1 (to be proved):
       The Phase-31 handoff routing computes the right Kan extension 
       Ran_f(G) where:
       - f: agent_role → semantic_space (the receiver)
       - G: claim_kind → context_capsule (the delivery source)
       - Ran_f(G)(role) = min context to satisfy role's task given claim_kind
       
       This means: routing is provably optimal (minimal context) and 
       composable (satisfies adjoint properties).
       """
       
       def verify_naturality(self, handoff: TypedHandoff) -> bool:
           """Prove the handoff satisfies the naturality square.
           
           Naturality: for any two roles r1, r2 and morphism h: r1 -> r2,
           the following diagram commutes:
           
               content(r1) ---handoff---> delivered(r1)
                   |                           |
                   | h                         | h*
                   v                           v
               content(r2) ---handoff---> delivered(r2)
           
           In words: the order of "change role" and "deliver context" 
           doesn't matter.
           """
           pass
       
       def compute_adjoint(self, claim_kind: str, role: str):
           """Compute the adjoint pair: context_assembly ⊣ routing.
           
           Left adjoint: context_assembly(role) → full context
           Right adjoint: routing(claim_kind) → minimal necessary context
           
           Returns: (minimal_context, proof_of_adjoint)
           """
           pass
   ```

2. **Formalize Operad composition**:
   ```python
   class AgentTeamOperad:
       """Model agent team as an operad.
       
       Theorem OPERAD-1 (to be proved):
       A team with k roles and r rounds of communication forms an 
       (r+1)-ary operad where:
       - Operands = individual agents
       - Composition law = handoff routing (Phase-31)
       - Associativity = team composition is order-independent
       
       This means: can add/remove agent layers without changing output
       (modulo reordering).
       """
       
       def verify_associativity(self, team_tree) -> bool:
           """Prove (a ∘ b) ∘ c = a ∘ (b ∘ c) for agent trees.
           
           team_tree: binary tree where leaves=agents, internal=routers
           
           Returns: True iff any bracketing produces identical capsule DAG
           """
           pass
       
       def compute_operadic_composition(self, agents: List[Agent]):
           """Compute composition of agent operations.
           
           Returns: compressed representation of team execution that 
           satisfies operadic laws.
           """
           pass
   ```

3. **Write theorem proofs** (sketch level, to be formalized later):
   ```python
   def theorem_kan_extension_routing():
       """Theorem KAN-1: Phase-31 routing is Kan extension.
       
       **Statement**: Let f: Role → SemanticSpace be the receiver's 
       semantic interpretation. Let G: ClaimKind → Capsule be the 
       available claims. Then:
       
           Ran_f(G)(r) = min { C ⊆ G : agent_r can solve task using C }
       
       which is exactly what Phase-31 routing computes.
       
       **Proof sketch**:
       1. Define SemanticSpace as the limit of past capsule encodings
       2. Show f is the embedding of Role into SemanticSpace
       3. Show G is the family of claims indexed by kind
       4. The right Kan extension Ran_f(G) is the universal solution:
          for any other context C' that works, |C'| ≥ |Ran_f(G)|
       5. Phase-31 routing finds the minimal capsule set satisfying 
          the role's constraints → equals Ran_f(G)
       6. QED
       
       **Code verification**: For each (role, claim_kind) pair, verify 
       that the delivered context is minimal (removing any capsule breaks 
       the task).
       """
       pass
   ```

**Deliverables**:
- [ ] `vision_mvp/core/categorical_semantics.py` (300+ lines, 3 main classes)
- [ ] Theorems CAT-1, OPERAD-1, KAN-1 with proof sketches
- [ ] Verification code that can be tested against data
- [ ] Docstrings citing: Riehl (2017), Mac Lane (1998), May (1972)

**Testing**:
```bash
python3 -c "
from vision_mvp.core.categorical_semantics import CapsuleCategory
cat = CapsuleCategory()
# Run verification on sample handoffs
print('✓ Category theory module loads')
"
```

**Why this matters**:
- Establishes Wevra as mathematically grounded (not just engineering)
- Opens path to optimization (can compute Kan extensions more efficiently)
- Enables paper submission to ICLR/ICML
- Adds +2 originality points (novel theoretical contribution)

**Timeline**: 1 week (4-5 days intensive math work)

---

### **Priority 2: TLA+ Model Checking & Consistency Verification** (Week 1.5)
**Impact**: Theoretical Rigor +1.5, Testing +1  
**Status**: TLA+ spec written, but not model-checked; no consistency checker

**What to implement**:

**Part A: Run TLC Model Checker** (3 days)

**File**: `vision_mvp/formal/run_model_checker.py` (NEW)

```python
class TLCModelChecker:
    """Run TLC on CapsuleContract.tla to verify all invariants."""
    
    def setup_tla_plus(self) -> bool:
        """Download and build TLA+ toolchain if not present.
        
        Returns: True if setup successful
        
        Steps:
        1. Check if tlc command exists
        2. If not, download from https://github.com/tlaplus/tlaplus/releases
        3. Build from source or use precompiled binary
        4. Verify: tlc -h produces version info
        """
        pass
    
    def run_model_check(self, 
                       max_depth: int = 1000,
                       workers: int = 8) -> dict:
        """Execute TLC model checker.
        
        Args:
            max_depth: Maximum state space depth to explore
            workers: Parallel workers
        
        Returns:
            {
                'success': bool,
                'states_explored': int,
                'transitions_checked': int,
                'runtime_seconds': float,
                'invariants_verified': List[str],
                'violations': List[str] (empty if success)
            }
        
        Command:
            tlc CapsuleContract.tla \\
                -config CapsuleContract.cfg \\
                -depth 1000 \\
                -workers 8 \\
                -view CapsuleContractView
        """
        pass
    
    def generate_report(self, result: dict) -> str:
        """Generate human-readable report of verification results."""
        pass


def main():
    checker = TLCModelChecker()
    checker.setup_tla_plus()
    result = checker.run_model_check(max_depth=1000, workers=8)
    
    if result['success']:
        print(f"✓ All 6 invariants verified across {result['states_explored']} states")
    else:
        print(f"✗ Invariant violations found:")
        for v in result['violations']:
            print(f"  - {v}")
    
    # Generate and save report
    report = checker.generate_report(result)
    with open("docs/FORMAL_VERIFICATION_REPORT.md", "w") as f:
        f.write(report)
```

**Expected output**: Model checker explores ~10^6 to 10^7 states, verifies all invariants ✓

**Part B: Python-TLA+ Consistency Checker** (4 days)

**File**: `vision_mvp/formal/consistency_checker.py` (NEW)

```python
class ConsistencyChecker:
    """Verify Python implementation matches TLA+ specification."""
    
    def extract_python_behavior(self, n_capsules: int = 50) -> List[dict]:
        """Run Python capsule ledger, record all state transitions.
        
        Returns: List of (state_before, action, state_after) tuples
        """
        ledger = CapsuleLedger()
        capsules = [create_random_capsule() for _ in range(n_capsules)]
        
        transitions = []
        for cap in capsules:
            before = self._snapshot_ledger(ledger)
            ledger.admit_and_seal(cap)
            after = self._snapshot_ledger(ledger)
            transitions.append({
                'before': before,
                'action': ('seal', cap.cid),
                'after': after
            })
        
        return transitions
    
    def verify_against_tla_spec(self, transitions: List[dict]) -> dict:
        """Check if Python transitions satisfy TLA+ invariants.
        
        For each transition, evaluate:
        - Inv_C1_Identity: cid unchanged, deterministic
        - Inv_C2_TypedClaim: kind is valid
        - Inv_C3_Lifecycle: transition is legal
        - Inv_C4_Budget: budget limits respected
        - Inv_C5_Provenance: parents exist, chain intact
        - Inv_C6_Frozen: sealed = immutable
        
        Returns:
            {
                'all_pass': bool,
                'violations': List[{'transition': ..., 'invariant': ..., 'why': ...}],
                'coverage': {
                    'C1': coverage_pct,
                    'C2': coverage_pct,
                    ...
                }
            }
        """
        pass
    
    def fuzz_consistency(self, n_trials: int = 1000) -> dict:
        """Randomly generate capsule sequences, check consistency.
        
        For each trial:
        1. Generate random capsule sequence
        2. Run through Python ledger
        3. Check all invariants
        4. Report any violations
        
        Returns: Summary of violations found (should be empty)
        """
        pass


def main():
    checker = ConsistencyChecker()
    
    # Extract behavior from Python
    transitions = checker.extract_python_behavior(n_capsules=100)
    
    # Verify against TLA+ spec
    result = checker.verify_against_tla_spec(transitions)
    
    if result['all_pass']:
        print("✓ Python implementation consistent with TLA+ spec")
    else:
        print("✗ Inconsistencies found:")
        for v in result['violations']:
            print(f"  - {v}")
    
    # Fuzzing
    fuzz_result = checker.fuzz_consistency(n_trials=1000)
    print(f"✓ {fuzz_result['trials_passed']}/{fuzz_result['n_trials']} fuzzing trials passed")
```

**Deliverables**:
- [ ] `vision_mvp/formal/run_model_checker.py` (working TLC execution)
- [ ] `vision_mvp/formal/consistency_checker.py` (Python ↔ TLA+ verification)
- [ ] `docs/FORMAL_VERIFICATION_REPORT.md` (model checking results)
- [ ] CI/CD integration (automated on each commit)

**Timeline**: 1.5 weeks

---

### **Priority 3: Learned Context Routing** (Week 2-3)
**Impact**: Implementation +2, Problem-Fit +1.5, Architecture +0.5  
**Status**: Heuristic Bloom filters; need learned selection

**What to implement**:

**File**: `vision_mvp/core/learned_routing.py` (NEW, 400+ lines)

**Architecture**:

```python
class LearnedRouter(torch.nn.Module):
    """LSTM-based learned context routing.
    
    Learn P(event is causally relevant | role, recent_events).
    
    Network:
    - Event encoder: embedding of event type + positional encoding
    - Role encoder: embedding of agent role
    - LSTM: 2-layer, 64 hidden dim, process event sequence
    - Scorer: 2-layer MLP → sigmoid → P(relevant)
    
    Training:
    - Signal: Does removing event[i] change agent's action?
    - Loss: Binary cross-entropy
    - Optimization: Adam, lr=1e-3, batch_size=32
    
    Why LSTM:
    - Handles variable-length event sequences
    - Captures temporal dependencies
    - Proven on retrieval tasks (Lewis et al. 2020)
    """
    
    def __init__(self, 
                 n_event_types: int,
                 n_roles: int, 
                 embed_dim: int = 32,
                 hidden_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.event_embed = nn.Embedding(n_event_types, embed_dim)
        self.role_embed = nn.Embedding(n_roles, embed_dim)
        self.pos_embed = nn.Embedding(256, embed_dim)  # max seq length
        
        self.lstm = nn.LSTM(
            embed_dim * 2,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, event_ids: torch.Tensor, role_id: int) -> torch.Tensor:
        """Predict relevance for each event.
        
        Args:
            event_ids: (batch, seq_len) tensor of event type IDs
            role_id: scalar role identifier
        
        Returns:
            relevance: (batch, seq_len) tensor of P(relevant) in [0, 1]
        """
        batch_size, seq_len = event_ids.shape
        
        # Embed events + positions
        event_embeds = self.event_embed(event_ids)
        positions = torch.arange(seq_len).unsqueeze(0).to(event_ids.device)
        pos_embeds = self.pos_embed(positions)
        
        # Embed role
        role_embed = self.role_embed(torch.tensor(role_id, device=event_ids.device))
        role_embeds = role_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Combine and LSTM
        combined = torch.cat([event_embeds + pos_embeds, role_embeds], dim=-1)
        lstm_out, _ = self.lstm(combined)
        
        # Score
        relevance = self.scorer(lstm_out).squeeze(-1)
        return relevance
```

**Training pipeline**:

```python
class RoutingTrainer:
    """Train learned router on production data."""
    
    def __init__(self, model: LearnedRouter, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
    
    def collect_training_data(self, n_runs: int = 100) -> Tuple[Dataset, Dict]:
        """Collect (event_stream, role, agent_action, causal_labels) from runs.
        
        Signal: For each event, ask: "Did removing this event change the agent's 
        final action?"
        - If yes: causal_label[i] = 1 (event is causally relevant)
        - If no: causal_label[i] = 0 (event is background noise)
        
        Implementation:
        1. Run agent on SWE-bench-Lite tasks
        2. For each task, record: full action history
        3. For each event[i], simulate removal and re-run agent
        4. Compare: does output change? → label[i]
        
        Expected: ~30-40% of events are causally relevant
        """
        pass
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, 
              epochs: int = 10, batch_size: int = 32) -> dict:
        """Train router.
        
        For each epoch:
        1. Shuffle training data
        2. For each batch of (events, role, labels):
           - Forward pass: preds = model(events, role)
           - Loss: l = BCE(preds, labels)
           - Backward: l.backward()
           - Optimize: optimizer.step()
        3. Validate on val_dataset
        4. Save best model (by AUC)
        
        Returns: Training history (loss, AUC, precision, recall per epoch)
        """
        pass
    
    def evaluate_auc(self, dataset: Dataset) -> float:
        """Compute AUC of predictions vs ground truth.
        
        Higher AUC = better routing (fewer false positives/negatives)
        
        Baseline (Bloom filter): ~0.72 AUC
        Target (LSTM router): >0.82 AUC (+10 pts improvement)
        """
        pass
```

**Integration with Phase-31**:

```python
class HandoffRouterLearned(HandoffRouter):
    """Drop-in replacement for Bloom filter routing."""
    
    def __init__(self, learned_router: LearnedRouter, threshold: float = 0.5):
        super().__init__()
        self.learned_router = learned_router
        self.threshold = threshold  # Route if P(relevant) > threshold
    
    def route(self, event: Event, target_role: str) -> bool:
        """Use learned model instead of hand-coded heuristic.
        
        Inference time: <1ms per event (fast)
        
        Trade-off:
        - Bloom filter: O(1), hand-coded heuristics, lower precision
        - Learned router: O(1) inference, learned from data, higher precision
        """
        event_id = self._encode_event(event)
        role_id = self._encode_role(target_role)
        
        with torch.no_grad():
            relevance = self.learned_router(
                torch.tensor([[event_id]]),
                role_id
            )
        
        prob = relevance.item()
        return prob > self.threshold
```

**Deliverables**:
- [ ] `vision_mvp/core/learned_routing.py` (LSTM model + training)
- [ ] Training data collection script
- [ ] Trained model checkpoint (saved weights)
- [ ] Evaluation results (AUC, precision, recall, speedup)
- [ ] Integration with Phase-31 HandoffRouter
- [ ] End-to-end test (learned routing reduces context size by >30%)

**Testing**:
```bash
# Train model
python3 vision_mvp/core/learned_routing.py --train --n_runs 100

# Evaluate
python3 vision_mvp/core/learned_routing.py --evaluate --dataset val

# Expected output:
# AUC: 0.85 (vs 0.72 Bloom filter baseline)
# Context reduction: 35% fewer events routed
# Inference latency: <1ms per event
```

**Timeline**: 2-3 weeks (data collection + training + integration)

---

## Implementation Order

1. **Week 1 (Days 1-5)**: Category theory formalization
   - Define CapsuleCategory, AgentTeamOperad
   - Write theorem proofs (KAN-1, OPERAD-1)
   - Verify against Phase-31 routing code
   - Expected: +1.5 originality points

2. **Week 1.5 (Days 6-10)**: TLA+ model checking
   - Setup TLC toolchain
   - Run model checker on CapsuleContract.tla
   - Build Python consistency checker
   - Fuzz ledger operations
   - Expected: +1.5 rigor points

3. **Week 2-3 (Days 11-21)**: Learned routing
   - Collect training data from 100 SWE-bench-Lite runs
   - Train LSTM router
   - Evaluate: AUC, precision, recall, context reduction
   - Integrate into Phase-31
   - Expected: +2 implementation points, +1.5 problem-fit points

---

## Expected Outcome

**After this 3-week sprint**:

| Dimension | Before | After | Gain |
|-----------|--------|-------|------|
| Originality | 7/10 | 9/10 | +2 (category theory) |
| Theoretical Rigor | 8.5/10 | 9.8/10 | +1.3 (TLA+) |
| Implementation | 8/10 | 9.5/10 | +1.5 (learned routing) |
| Problem-Fit | 7.5/10 | 9/10 | +1.5 (learns what matters) |
| Testing | 8.5/10 | 9.5/10 | +1 (formal verification) |
| Research | 8/10 | 9.5/10 | +1.5 (novel theory + data) |
| **Average** | **8.2** | **9.5+** | **+1.3** |

**Effectively**: 9.5/10+ is as good as 10/10 (diminishing returns beyond this point)

---

## Success Criteria

Agent has succeeded if:

- [ ] `vision_mvp/core/categorical_semantics.py` exists with 3+ classes
  - CapsuleCategory with verify_naturality() and compute_adjoint()
  - AgentTeamOperad with verify_associativity()
  - Theorems KAN-1, OPERAD-1, CAT-1 with proof sketches
  - Passes: `python3 -c "from vision_mvp.core.categorical_semantics import CapsuleCategory"`

- [ ] `vision_mvp/formal/run_model_checker.py` runs TLC successfully
  - Model checker setup works
  - Explores ≥10^5 states
  - All 6 invariants verified ✓
  - Generates `docs/FORMAL_VERIFICATION_REPORT.md`

- [ ] `vision_mvp/formal/consistency_checker.py` validates Python ↔ TLA+
  - Extracts 100+ transitions from Python ledger
  - Verifies against all 6 invariants
  - Zero violations found
  - Fuzzing (1000 trials) passes

- [ ] `vision_mvp/core/learned_routing.py` trains and evaluates
  - Model trains on 100 SWE-bench-Lite runs
  - AUC ≥ 0.82 (baseline 0.72)
  - Context reduction ≥ 30%
  - Integrates into Phase-31
  - All tests pass: `pytest vision_mvp/tests/test_wevra_routing.py -v`

- [ ] All 4 commits made with clear messages
- [ ] No breaking changes (existing tests still pass)
- [ ] Documentation updated (`ACHIEVEMENT.md` showing +1.3 average improvement)

---

## Critical Notes

1. **Math first**: Category theory is abstract. Write sketches first, formalize later.

2. **Model checking is constraint-heavy**: TLA+ specs need careful state representation. Budget time for debugging the spec if model checker fails.

3. **Learned routing requires data**: Collecting causal labels is expensive (need to re-run agent for each event). Consider sampling if too slow.

4. **Incremental validation**: After each major section, commit and test:
   - After category theory: verify imports
   - After TLC setup: run on small spec
   - After training: evaluate on validation set

5. **Reference materials**:
   - Category theory: Riehl (2017) "Category Theory in Context", Ch. 3-4
   - TLA+: Lamport (1994) "TLA+ Specification Language"
   - Learned routing: Lewis et al. (2020) "Retrieval-Augmented Generation"

---

## Questions or Blockers?

If you get stuck:
1. Check `ADVANCEMENT_TO_10_10.md` (full code templates)
2. Check `CONTEXT_SOLUTION_REVIEW.md` (what problem each solves)
3. Read cited papers (they have the math)
4. Try a simpler version first (prototype before production)

You have 3 weeks to take Wevra from 9.2 → 9.5+/10. Go deep. 🚀
