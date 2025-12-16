# Migration from Reflective Agent to Entity Refinement Agent

## Overview

The system has been **completely refactored** to replace the complex reflective agent with a simpler, more focused **entity-based refinement approach**. This change simplifies the architecture and makes the refinement process more transparent and effective.

---

## What Changed?

### **Before: Reflective Agent**
- Used LLM-based critique with multiple quality criteria (accuracy, completeness, naturalness, conciseness)
- Required dual approval: both reflective agent AND entity verification had to pass
- Complex scoring system with weighted metrics
- Adaptive iteration strategies based on code complexity
- Entity verification was a secondary check

### **After: Entity Refinement Agent**
- **Single source of truth**: Entity verification is the ONLY criterion for approval
- **Instruction-based feedback**: When entity verification fails, the instruction agent generates specific, actionable feedback
- **Direct refinement**: Feedback is fed directly to the model for regeneration
- **Simpler logic**: No dual approval, no complex scoring, just entity metrics (precision, recall, F1, hallucination score)
- **Transparent**: Clear logging of what entities are hallucinated or missing

---

## Architecture Changes

### **New Component: `EntityRefinementAgent`**
Location: `src/agent/entity_refinement_agent.py`

**Key Features:**
1. **Entity Verification Loop**
   - Verifies entities in generated summary
   - Checks: function names, parameters, called functions, return types, variables
   
2. **Instruction-Based Feedback**
   - Uses `InstructionAgent` to generate specific feedback when verification fails
   - Feedback includes:
     - ❌ Hallucinated entities to remove
     - ➕ Missing entities to include
     - 📝 Specific guidance (e.g., "mention all parameters")
   
3. **Iterative Refinement**
   - Model receives combined feedback and regenerates summary
   - Continues until entity verification passes OR max iterations reached
   
4. **Stopping Criteria**
   - ✓ Entity verification passes (hallucination score below threshold)
   - ⚠ Summary converges (no changes between iterations)
   - ⚠ Max iterations reached

### **Removed Component: `ReflectiveAgent`**
Location: `src/agent/reflective_agent.py` (still exists but no longer used)

**What was removed:**
- LLM-based critique generation
- Multi-criteria scoring (accuracy, completeness, naturalness, conciseness)
- Dual approval mechanism
- Adaptive iteration strategies
- Complex approval logic with keyword matching

---

## Updated Files

### **1. Core Agent**
- ✅ **Created**: `src/agent/entity_refinement_agent.py`
- ⚠️ **Deprecated**: `src/agent/reflective_agent.py` (no longer used)

### **2. Inference Pipeline**
- ✅ **Updated**: `src/model/inference.py`
  - Changed import from `ReflectiveAgent` → `EntityRefinementAgent`
  - Renamed parameter: `reflective_agent` → `entity_agent`
  - Renamed flag: `use_reflective_agent` → `use_entity_agent`

### **3. Evaluation Script**
- ✅ **Updated**: `evaluate.py`
  - Changed import from `ReflectiveAgent` → `EntityRefinementAgent`
  - Renamed CLI flag: `--no_reflective_agent` → `--no_entity_agent`
  - Updated config path for max_iterations: `reflective_agent` → `entity_verification`

### **4. Inference Script**
- ✅ **Updated**: `run_inference.py`
  - Changed import from `ReflectiveAgent` → `EntityRefinementAgent`
  - Renamed CLI flag: `--no_reflective_agent` → `--no_entity_agent`
  - Updated all variable references

### **5. Configuration**
- ✅ **Updated**: `config.yaml`
  - Added entity refinement agent settings to `entity_verification` section:
    - `max_iterations`: 3
    - `max_iterations_eval`: 3
    - `temperature`: 0.7
    - `max_tokens_refinement`: 300
    - `fast_mode`: false
    - `greedy_decoding`: false
  - Kept `reflective_agent` section for backward compatibility (but it's no longer used)

---

## How It Works Now

### **Workflow:**

```
1. Generate initial summary
   ↓
2. Run entity verification
   ↓
3. Verification passed? → ✓ DONE
   ↓ No
4. Generate instruction-based feedback
   - "Remove hallucinated entity: 'foo'"
   - "Add missing parameter: 'bar'"
   ↓
5. Feed feedback to model → Regenerate summary
   ↓
6. Repeat from step 2 (max 3 iterations)
```

### **Example Feedback:**

```
⚠️ Entity Verification Failed (Hallucination Score: 0.45 > 0.30)

❌ Remove these hallucinated entities (they don't exist in the code):
  - 'process_data'
  - 'validate_input'

➕ Include these missing entities in the description:
  - 'filename'
  - 'encoding'

📋 Action Required:
Rewrite the docstring to:
1. Remove references to: process_data, validate_input
2. Add descriptions for: filename, encoding
3. Ensure accuracy - only describe what the code actually does
```

---

## Benefits

### **1. Simplicity**
- ✅ Single criterion: entity verification
- ✅ No complex scoring or approval logic
- ✅ Easier to understand and debug

### **2. Transparency**
- ✅ Clear feedback on what's wrong (hallucinated vs. missing entities)
- ✅ Detailed logging of entity metrics (precision, recall, F1)
- ✅ Easy to track improvements across iterations

### **3. Effectiveness**
- ✅ Directly addresses hallucination problem
- ✅ Ensures all parameters are mentioned
- ✅ Prevents model from inventing non-existent functions

### **4. Maintainability**
- ✅ Fewer moving parts
- ✅ Single source of truth for quality
- ✅ Easier to tune (just adjust entity weights and threshold)

---

## Configuration

### **Entity Verification Settings** (`config.yaml`)

```yaml
entity_verification:
  enabled: true
  
  # Iteration settings
  max_iterations: 3
  max_iterations_eval: 3
  
  # Generation settings
  temperature: 0.7
  max_tokens_refinement: 300
  fast_mode: false
  greedy_decoding: false
  
  # Entity verification thresholds
  hallucination_threshold: 0.30  # Max acceptable hallucination score
  entity_weights:
    function_names: 1.0      # Critical
    parameter_names: 1.0     # Critical
    called_functions: 0.7    # Important
    return_types: 0.7        # Important
    variables: 0.3           # Optional
  require_all_params: true
  allow_synonyms: true
```

---

## Usage

### **Evaluation:**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-xxx
```

### **Disable Entity Agent:**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-xxx --no_entity_agent
```

### **Fast Mode:**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-xxx --fast_mode
```

### **Custom Iterations:**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-xxx --max_iterations 5
```

### **Inference:**
```bash
python run_inference.py --checkpoint ./outputs/checkpoint-xxx --code "def add(a, b): return a + b"
```

---

## Backward Compatibility

- ⚠️ The `reflective_agent` section in `config.yaml` is **no longer used** but kept for reference
- ⚠️ Old scripts using `ReflectiveAgent` will need to be updated
- ⚠️ Test files like `test_reflective_agent_fix.py` may need updates

---

## Next Steps

1. ✅ **Test the new entity refinement agent** with a small dataset
2. ✅ **Compare results** with the old reflective agent approach
3. ✅ **Update documentation** and README
4. ✅ **Remove or archive** old reflective agent code if no longer needed
5. ✅ **Update test files** to use the new agent

---

## Summary

The migration from **Reflective Agent** to **Entity Refinement Agent** represents a **significant simplification** of the refinement process. By focusing solely on entity verification and using instruction-based feedback, the system becomes more transparent, maintainable, and effective at preventing hallucinations while ensuring completeness.

**Key Takeaway**: Instead of asking the LLM to critique itself on multiple vague criteria, we now use **concrete entity verification** to provide **specific, actionable feedback** that directly addresses the core problem: hallucinations and missing information.
