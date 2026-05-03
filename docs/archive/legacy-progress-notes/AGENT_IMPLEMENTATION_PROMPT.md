# Implementation Prompt for Agent: 1-Day Sprint to 10/10

## Context
You are continuing work on **CoordPy: Context-Capsule Runtime** (Context Zero research programme). The codebase currently scores **8.2/10 average** across 10 dimensions. Your mission is to implement **quick-win improvements** to reach **9.5+/10** in a single 8-hour working day.

**Reference document**: `ADVANCEMENT_TO_10_10.md` (in same repo). Read it first for full context.

---

## Mission: Implement 4 Quick-Win Systems

You will implement these in order (priority by impact × effort):

### **Priority 1: Property-Based Testing Framework** (Testing +1, ~3 hours)
**File**: `vision_mvp/tests/test_capsule_properties.py` (NEW)
**What it does**: Auto-generate 1000s of test cases using Hypothesis library to catch edge cases.
**Success criteria**: 
- [ ] File created with ≥5 properties (@given decorated functions)
- [ ] Each property tests a contract invariant (C1, C2, C3, C4, C5, C6)
- [ ] Runs without errors: `pytest vision_mvp/tests/test_capsule_properties.py -v`
- [ ] Generates ≥100 test cases per property (default Hypothesis setting)

**Key properties to implement** (copy from `ADVANCEMENT_TO_10_10.md` Part III, Section 1):
1. `test_c1_identity_deterministic` — cid(cap) = cid(cap) always
2. `test_c2_unknown_kind_rejected` — only closed-vocabulary kinds allowed
3. `test_c3_lifecycle_order` — transitions must be PROPOSED → ADMITTED → SEALED
4. `test_c4_budget_monotonic` — if cap fits budget, variant with +tokens still fits
5. `test_metamorphic_removal` — removing event never grows context size

**Code template**: See `ADVANCEMENT_TO_10_10.md` Part III.1 for full code. Copy and adapt.

**Dependencies**: 
```bash
pip install hypothesis pytest
```

---

### **Priority 2: Layered API Design** (Usability +1, ~2 hours)
**File**: `vision_mvp/coordpy/api_layers.py` (NEW)
**What it does**: Create 3-tier API (high/mid/low level) so users at different skill levels can use CoordPy.
**Success criteria**:
- [ ] File created with 3 classes: `CoordPySimpleAPI`, `CoordPyBuilderAPI`, `CoordPyAdvancedAPI`
- [ ] Each class has docstrings explaining intended audience
- [ ] Imports check: `from vision_mvp.coordpy.api_layers import CoordPySimpleAPI`
- [ ] Example usage works: can instantiate each class without errors

**High-level (for end users)**:
```python
from coordpy import run_profile
result = run_profile("bundled_57", team_size=10)
```

**Mid-level (for developers)**:
```python
from coordpy.api_layers import CoordPyBuilderAPI
config = (CoordPyBuilderAPI()
    .with_team_size(10)
    .with_role("code_writer", budget=4096)
    .with_compression("medium")
    .build())
```

**Low-level (for researchers)**:
```python
from coordpy.core import CapsuleLedger, HandoffRouter
ledger = CapsuleLedger()
router = HandoffRouter(subscription_table={...})
```

**Code template**: See `ADVANCEMENT_TO_10_10.md` Part III.4 for full code.

---

### **Priority 3: TLA+ Formal Specification** (Theoretical Rigor +1, ~2.5 hours)
**File**: `vision_mvp/formal/CapsuleContract.tla` (NEW)
**What it does**: Machine-executable formal spec of the 6 contract invariants (C1-C6).
**Success criteria**:
- [ ] File created with valid TLA+ syntax
- [ ] Defines all 6 invariants: `Inv_C1_Identity`, `Inv_C2_TypedClaim`, ... `Inv_C6_Frozen`
- [ ] File parses without syntax errors (TLA+ syntax checker if available)
- [ ] Contains at least one `INVARIANT` statement per contract clause

**Why this matters**: TLA+ makes the contract executable. Can be model-checked later (outside 1-day scope).

**Code template**: See `ADVANCEMENT_TO_10_10.md` Part I.2 for full TLA+ spec.

**Key structure**:
```tla
(* Module CapsuleContract *)
EXTENDS Naturals, Sequences, FiniteSets

VARIABLE ledger, pending, chain_hashes

Inv_C1_Identity == \A c \in ledger: c.cid = SHA256(...)
Inv_C2_TypedClaim == \A c \in ledger \cup pending: c.kind \in {...}
Inv_C3_Lifecycle == \A c \in ledger: c.lifecycle \in {...}
...
AllInvariants == Inv_C1_Identity /\ Inv_C2_TypedClaim /\ ...
```

**Note**: Don't need to run TLC model checker today. Just create the spec file.

---

### **Priority 4: Theorem Auto-Documentation** (Documentation +0.5, ~1.5 hours)
**File**: `vision_mvp/scripts/generate_theorem_docs.py` (NEW)
**What it does**: Scan codebase for "Theorem" declarations and auto-generate markdown documentation.
**Success criteria**:
- [ ] Script created and runs without errors: `python vision_mvp/scripts/generate_theorem_docs.py`
- [ ] Output file `docs/THEOREMS_AUTO.md` generated
- [ ] Auto-doc contains ≥5 theorem entries (scan existing code for Theorem declarations)
- [ ] Each entry has: theorem ID, statement, source file reference

**Code template**: See `ADVANCEMENT_TO_10_10.md` Part III.3 for full code.

**How it works**:
1. Use `ast` module to parse Python files in `vision_mvp/`
2. Find functions starting with `theorem_`
3. Extract docstrings containing "Theorem W<N>-<TYPE>-<N>"
4. Generate markdown with cross-references

---

## Implementation Order & Time Breakdown

| Priority | Task | Duration | Running Total |
|----------|------|----------|----------------|
| 1 | Property tests setup + 5 properties | 3 hours | 3h |
| 2 | Layered API (3 tiers) | 2 hours | 5h |
| 3 | TLA+ spec file | 2.5 hours | 7.5h |
| 4 | Theorem auto-doc | 1.5 hours | 9h |
| — | **Padding (testing, debugging)** | **1 hour** | **~8h effective** |

**Total**: Fits in 8-hour workday with 1 hour buffer for debugging/integration.

---

## Detailed Instructions Per Task

### Task 1: Property-Based Testing (3 hours)

**Step 1.1**: Create file
```bash
touch vision_mvp/tests/test_capsule_properties.py
```

**Step 1.2**: Copy code template from `ADVANCEMENT_TO_10_10.md` Part III, Section 1
- Start with imports (hypothesis, pytest)
- Define `strategy_capsule()` — generates random Capsule objects
- Implement 5 property tests (see list above)

**Step 1.3**: Run tests
```bash
cd /Users/qdong/Desktop/Desktop1/Andy/context-zero
pytest vision_mvp/tests/test_capsule_properties.py -v
```

**Step 1.4**: Verify output
- Should see "5 passed" (or similar)
- No import errors
- Hypothesis shows number of examples generated (target: ≥100 per property)

**Deliverable**: Working test file with green test run output.

---

### Task 2: Layered API (2 hours)

**Step 2.1**: Create file
```bash
touch vision_mvp/coordpy/api_layers.py
```

**Step 2.2**: Implement 3 classes
```python
class CoordPySimpleAPI:
    """High-level: for end users who just want to run profiles."""
    def run_profile(self, profile_name: str, team_size: int):
        """Simple entry point."""
        pass

class CoordPyBuilderAPI:
    """Mid-level: for developers building custom configs."""
    def __init__(self):
        self.config = {}
    def with_team_size(self, n: int):
        self.config['team_size'] = n
        return self
    def with_role(self, role: str, budget: int):
        # Store role config
        return self
    def build(self):
        """Return CoordPyConfig."""
        pass

class CoordPyAdvancedAPI:
    """Low-level: for researchers accessing substrate."""
    def __init__(self):
        from vision_mvp.core import CapsuleLedger, HandoffRouter
        self.ledger = CapsuleLedger()
        self.router = HandoffRouter()
```

**Step 2.3**: Add to `vision_mvp/coordpy/__init__.py`
```python
from vision_mvp.coordpy.api_layers import (
    CoordPySimpleAPI,
    CoordPyBuilderAPI,
    CoordPyAdvancedAPI
)
__all__ = [..., 'CoordPySimpleAPI', 'CoordPyBuilderAPI', 'CoordPyAdvancedAPI']
```

**Step 2.4**: Test imports
```python
from vision_mvp.coordpy import CoordPySimpleAPI, CoordPyBuilderAPI, CoordPyAdvancedAPI
api_simple = CoordPySimpleAPI()
api_builder = CoordPyBuilderAPI().with_team_size(10).build()
api_advanced = CoordPyAdvancedAPI()
print("✓ All 3 API tiers import successfully")
```

**Deliverable**: File `api_layers.py` with 3 classes, clean imports.

---

### Task 3: TLA+ Specification (2.5 hours)

**Step 3.1**: Create directory and file
```bash
mkdir -p vision_mvp/formal
touch vision_mvp/formal/CapsuleContract.tla
```

**Step 3.2**: Copy TLA+ template from `ADVANCEMENT_TO_10_10.md` Part I, Section 2
- Module header: `EXTENDS Naturals, Sequences, FiniteSets`
- Define state variables: `VARIABLE ledger, pending, chain_hashes`
- Copy all 6 invariant definitions (Inv_C1 through Inv_C6)
- Define `AllInvariants` conjunction

**Step 3.3**: Syntax validation (optional, if TLA+ tools available)
```bash
# If tlc installed:
cd vision_mvp/formal
tlc CapsuleContract.tla -check  # Just check syntax, don't run
```

If TLA+ tools not available, just verify it reads without error:
```bash
cat vision_mvp/formal/CapsuleContract.tla | head -50
```

**Deliverable**: Valid TLA+ file (`CapsuleContract.tla`) with all 6 invariants defined.

---

### Task 4: Theorem Auto-Documentation (1.5 hours)

**Step 4.1**: Create script
```bash
touch vision_mvp/scripts/generate_theorem_docs.py
```

**Step 4.2**: Copy code from `ADVANCEMENT_TO_10_10.md` Part III, Section 3
- `TheoremExtractor(ast.NodeVisitor)` class
- `extract_all_theorems()` function to scan directory
- `generate_theorem_documentation()` to render markdown
- `main()` entry point

**Step 4.3**: Run script
```bash
cd /Users/qdong/Desktop/Desktop1/Andy/context-zero
python vision_mvp/scripts/generate_theorem_docs.py
```

**Step 4.4**: Verify output
```bash
ls -lh docs/THEOREMS_AUTO.md
head -50 docs/THEOREMS_AUTO.md
```

Should show:
- File created ✓
- Contains theorem entries ✓
- Has proper markdown formatting ✓

**Deliverable**: Script that generates `docs/THEOREMS_AUTO.md` with ≥5 theorem entries.

---

## Integration Checklist

After all 4 tasks, verify everything works together:

```bash
# 1. Tests run
pytest vision_mvp/tests/test_capsule_properties.py -v

# 2. API imports work
python -c "from vision_mvp.coordpy import CoordPySimpleAPI; print('✓')"

# 3. TLA+ file exists and is valid
test -f vision_mvp/formal/CapsuleContract.tla && echo "✓"

# 4. Auto-doc generated
test -f docs/THEOREMS_AUTO.md && echo "✓"

# 5. No import errors in main package
python -c "import vision_mvp.coordpy; print('✓')"
```

All should pass ✓.

---

## Expected Outcome

**After 1 day of implementation:**

- **Testing dimension**: 8.5 → 9.5 (property tests catch edge cases)
- **Usability dimension**: 7 → 8.5 (3-tier API meets users where they are)
- **Theoretical Rigor**: 8.5 → 9.5 (TLA+ spec makes contract formal)
- **Documentation**: 8 → 8.5 (auto-doc keeps proofs in sync with code)

**Average improvement**: +1 point → from 8.2 to 9.2/10

**Deliverables to commit**:
```
+ vision_mvp/tests/test_capsule_properties.py
+ vision_mvp/coordpy/api_layers.py
+ vision_mvp/formal/CapsuleContract.tla
+ vision_mvp/scripts/generate_theorem_docs.py
+ docs/THEOREMS_AUTO.md (auto-generated)
- vision_mvp/coordpy/__init__.py (modified: add API tiers to exports)
```

---

## Critical Notes

1. **Order matters**: Do Priority 1 first (tests), then 2-4. Don't jump around.

2. **Code quality**: Copy from `ADVANCEMENT_TO_10_10.md` templates exactly. Don't improvise; templates are research-backed.

3. **Testing as you go**: After each task, run the quick test:
   ```bash
   python -c "from vision_mvp.coordpy.api_layers import CoordPySimpleAPI; print('✓')"
   ```

4. **Commit per task**: After each of 4 tasks, create a git commit:
   ```bash
   git add vision_mvp/tests/test_capsule_properties.py
   git commit -m "Add property-based tests for Capsule invariants (C1-C6)
   
   - 5 properties testing contract invariants
   - Hypothesis generates 1000+ test cases
   - All C1-C6 covered
   
   Generated with [Claude Code](https://claude.ai/code)
   via [Happy](https://happy.engineering)"
   ```

5. **If stuck**: Refer back to `ADVANCEMENT_TO_10_10.md` sections cited. All code is there.

---

## Success Criteria (End of Day)

Agent has succeeded if:

- [ ] `vision_mvp/tests/test_capsule_properties.py` exists and pytest passes
- [ ] `vision_mvp/coordpy/api_layers.py` exists with 3 working classes
- [ ] `vision_mvp/formal/CapsuleContract.tla` exists with all 6 invariants
- [ ] `vision_mvp/scripts/generate_theorem_docs.py` runs and generates `docs/THEOREMS_AUTO.md`
- [ ] All 4 files committed to git with clear commit messages
- [ ] No import errors in main package
- [ ] Documentation updated with new features

**Bonus** (if time permits, <1 hour left):
- [ ] Run full test suite: `pytest vision_mvp/tests/ -v` (shouldn't break anything)
- [ ] Add a quick README entry pointing to new API tiers
- [ ] Create a simple example notebook showing high/mid/low-level usage

---

## Reference: ADVANCEMENT_TO_10_10.md Structure

If you need to find code templates, look at:
- **Part I, Section 1**: Category Theory (skip, 1-day out of scope)
- **Part I, Section 2**: TLA+ spec ← Use for Task 3
- **Part III, Section 1**: Property tests ← Use for Task 1
- **Part III, Section 3**: Auto-doc ← Use for Task 4
- **Part III, Section 4**: API design ← Use for Task 2

All code is there; copy and adapt minimally.

---

## Questions?

If anything is unclear:
1. Check `ADVANCEMENT_TO_10_10.md` for full context
2. Read the relevant section (cited above)
3. Copy the code template
4. Adapt to CoordPy codebase (file paths, imports)
5. Test it

You have a 1-day sprint. Go! 🚀
