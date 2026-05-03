You are an AI agent. Your mission: advance CoordPy (context-capsule runtime) from 9.2/10 to 9.5+/10 in 3 weeks.

Current status: Property tests, layered API, TLA+ spec written. Need: mathematical depth + formal verification + learned routing.

PRIORITY 1: CATEGORY THEORY (Week 1, 5 days)

Create file: vision_mvp/core/categorical_semantics.py (300 lines)

Add class CapsuleCategory:
```python
class CapsuleCategory:
    """Symmetric monoidal category of capsules.
    
    Theorem KAN-1: Phase-31 routing computes Kan extensions.
    - f = receiver role's semantic space
    - G = available claims by kind
    - Ran_f(G) = minimal context satisfying role
    
    Prove this: routing is provably optimal.
    """
    
    def verify_naturality(self, handoff: TypedHandoff) -> bool:
        """Prove handoff is natural transformation.
        
        For roles r1, r2 and morphism h: r1->r2, verify:
        context(r1) --handoff--> delivered(r1)
             |h                       |h*
             v                        v
        context(r2) --handoff--> delivered(r2)
        
        Should commute (order of role change + delivery doesn't matter).
        """
        # Implement: For each pair of roles, verify commutative diagram
        pass
    
    def compute_adjoint(self, claim_kind: str, role: str):
        """Compute adjoint: context_assembly ⊣ routing.
        
        Left adjoint = full context
        Right adjoint = minimal necessary context
        """
        pass
```

Add class AgentTeamOperad:
```python
class AgentTeamOperad:
    """Model team as operad (algebra of composition).
    
    Theorem OPERAD-1: Team with k roles + r rounds forms (r+1)-ary operad.
    - Composition law = handoff routing
    - Associativity = order-independent
    
    Prove this: can add/remove agent layers, output unchanged.
    """
    
    def verify_associativity(self, team_tree) -> bool:
        """Prove (a ∘ b) ∘ c = a ∘ (b ∘ c) for agents.
        
        team_tree = binary tree (leaves=agents, internals=routers)
        
        Verify all bracketings produce identical capsule DAG.
        """
        pass
```

Test: python3 -c "from vision_mvp.core.categorical_semantics import CapsuleCategory; print('✓')"

Commit: "Add categorical semantics: Kan extensions + operad composition (Theorems KAN-1, OPERAD-1)"

---

PRIORITY 2: TLA+ MODEL CHECKING (Week 1.5, 5 days)

Create file: vision_mvp/formal/run_model_checker.py

```python
class TLCModelChecker:
    """Run TLC model checker on CapsuleContract.tla."""
    
    def setup_tla_plus(self) -> bool:
        """Install TLA+ if not present.
        
        Download: https://github.com/tlaplus/tlaplus/releases
        Verify: tlc -h works
        """
        pass
    
    def run_model_check(self, max_depth=1000, workers=8) -> dict:
        """Execute: tlc CapsuleContract.tla -depth 1000 -workers 8
        
        Returns: {
            'success': bool,
            'states_explored': int,
            'invariants_verified': ['C1','C2','C3','C4','C5','C6'],
            'violations': []
        }
        
        Expected: ✓ All 6 invariants verified, 10^6+ states explored.
        """
        # Run subprocess: tlc CapsuleContract.tla -depth 1000
        # Parse output, extract results
        pass
```

Create file: vision_mvp/formal/consistency_checker.py

```python
class ConsistencyChecker:
    """Verify Python implementation matches TLA+ spec."""
    
    def extract_python_behavior(self, n_capsules=100) -> List[dict]:
        """Run Python ledger, record state transitions."""
        from vision_mvp.coordpy.capsule import CapsuleLedger, ContextCapsule, CapsuleBudget, CapsuleKind
        
        ledger = CapsuleLedger()
        transitions = []
        
        for _ in range(n_capsules):
            # Generate random capsule
            cap = ContextCapsule.new(
                kind=CapsuleKind.ARTIFACT,
                payload={'test': 'data'},
                budget=CapsuleBudget(max_bytes=1024),
                parents=()
            )
            
            before = self._snapshot(ledger)
            ledger.admit_and_seal(cap)
            after = self._snapshot(ledger)
            
            transitions.append({'before': before, 'action': ('seal', cap.cid), 'after': after})
        
        return transitions
    
    def _snapshot(self, ledger):
        """Snapshot ledger state."""
        return {'size': len(ledger._by_cid), 'chain_ok': ledger.verify_chain()}
    
    def verify_against_tla_spec(self, transitions) -> dict:
        """Check if transitions satisfy all 6 TLA+ invariants.
        
        For each transition, verify:
        - C1: cid deterministic
        - C2: kind in vocab
        - C3: lifecycle legal
        - C4: budget enforced
        - C5: chain intact
        - C6: sealed immutable
        
        Returns: {'all_pass': bool, 'violations': []}
        """
        # For each transition, check invariants
        violations = []
        
        for trans in transitions:
            # Check C1, C2, C3, C4, C5, C6
            # Append to violations if failed
            pass
        
        return {'all_pass': len(violations) == 0, 'violations': violations}
    
    def fuzz_consistency(self, n_trials=1000) -> dict:
        """Generate 1000 random capsule sequences, verify zero violations."""
        from vision_mvp.coordpy.capsule import CapsuleLedger, ContextCapsule, CapsuleBudget, CapsuleKind
        
        for trial in range(n_trials):
            ledger = CapsuleLedger()
            
            # Random operations
            for _ in range(10):
                cap = ContextCapsule.new(
                    kind=CapsuleKind.ARTIFACT,
                    payload={'trial': trial},
                    budget=CapsuleBudget(max_bytes=1024),
                    parents=()
                )
                ledger.admit_and_seal(cap)
            
            # Verify all invariants
            assert ledger.verify_chain(), f"Trial {trial} failed"
        
        return {'trials_passed': n_trials, 'n_trials': n_trials}
```

Test:
```bash
python3 vision_mvp/formal/run_model_checker.py
python3 vision_mvp/formal/consistency_checker.py
```

Commit: "Add formal verification: TLC model checking + consistency checker (0 violations)"

---

PRIORITY 3: LEARNED ROUTING (Week 2-3, 10 days)

Create file: vision_mvp/core/learned_routing.py (400 lines)

```python
import torch
import torch.nn as nn

class LearnedRouter(nn.Module):
    """LSTM predicts P(event causally relevant | role, events)."""
    
    def __init__(self, n_event_types: int, n_roles: int, 
                 embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        
        self.event_embed = nn.Embedding(n_event_types, embed_dim)
        self.role_embed = nn.Embedding(n_roles, embed_dim)
        self.pos_embed = nn.Embedding(256, embed_dim)
        
        self.lstm = nn.LSTM(embed_dim * 2, hidden_dim, 
                            num_layers=2, batch_first=True, dropout=0.1)
        
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, event_ids: torch.Tensor, role_id: int) -> torch.Tensor:
        """Predict relevance for each event (batch, seq_len)."""
        batch_size, seq_len = event_ids.shape
        
        # Embed
        event_embeds = self.event_embed(event_ids)
        positions = torch.arange(seq_len, device=event_ids.device).unsqueeze(0)
        pos_embeds = self.pos_embed(positions)
        role_embed = self.role_embed(torch.tensor(role_id, device=event_ids.device))
        
        # Combine
        combined = torch.cat([
            event_embeds + pos_embeds,
            role_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        ], dim=-1)
        
        # LSTM + score
        lstm_out, _ = self.lstm(combined)
        relevance = self.scorer(lstm_out).squeeze(-1)
        
        return relevance


class RoutingTrainer:
    """Train router on SWE-bench-Lite data."""
    
    def __init__(self, model: LearnedRouter, lr: float = 1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, events: torch.Tensor, roles: torch.Tensor, 
                    labels: torch.Tensor) -> float:
        """Train one epoch."""
        self.optimizer.zero_grad()
        
        preds = self.model(events, roles[0].item())
        loss = self.criterion(preds, labels.float())
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, events: torch.Tensor, roles: torch.Tensor, 
                 labels: torch.Tensor) -> dict:
        """Evaluate on test set.
        
        Returns: {'auc': float, 'precision': float, 'recall': float}
        """
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        with torch.no_grad():
            preds = self.model(events, roles[0].item()).cpu().numpy()
        
        labels = labels.cpu().numpy()
        
        auc = roc_auc_score(labels, preds)
        precision = precision_score(labels, (preds > 0.5).astype(int))
        recall = recall_score(labels, (preds > 0.5).astype(int))
        
        return {'auc': auc, 'precision': precision, 'recall': recall}
```

Usage:
```python
# Create model
router = LearnedRouter(n_event_types=100, n_roles=10)
trainer = RoutingTrainer(router)

# Train (on synthetic data for now)
events = torch.randint(0, 100, (32, 50))  # (batch, seq_len)
roles = torch.randint(0, 10, (32,))
labels = torch.randint(0, 2, (32, 50)).float()

for epoch in range(10):
    loss = trainer.train_epoch(events, roles, labels)
    print(f"Epoch {epoch}: loss={loss:.4f}")

# Evaluate
results = trainer.evaluate(events, roles, labels)
print(f"AUC: {results['auc']:.2f}")
```

Test: python3 -c "from vision_mvp.core.learned_routing import LearnedRouter; print('✓')"

Commit: "Add learned routing: LSTM context selection (trainable, 0.85 AUC)"

---

VERIFICATION CHECKLIST

After 3 weeks, verify ALL:

- [ ] vision_mvp/core/categorical_semantics.py exists, imports work
- [ ] TLC model checker runs, verifies 6 invariants on 10^6+ states
- [ ] ConsistencyChecker finds 0 violations in 1000 fuzzing trials
- [ ] LearnedRouter trains, evaluates with AUC > 0.80
- [ ] 3 commits made with clear messages
- [ ] Existing tests still pass (no breaking changes)
- [ ] Average score improved: 9.2 → 9.5+

Expected outcome: CoordPy at 9.5+/10 (effectively 10/10)
