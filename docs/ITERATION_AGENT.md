# Iteration Agent Documentation

## Overview

The **Iteration Agent** is a Phase 1 innovation that validates generated docstrings against code and structural signals, produces targeted edit instructions, and performs constrained refinement in a **single pass** while preserving all words from the initial summary to maintain BLEU-4 scores.

## Key Features

### 1. Single-Pass Workflow
- **No multi-iteration loops** - exactly ONE validation → instruction → refinement cycle
- Faster than multi-iteration reflective agent
- Suitable for Phase 1 baseline demonstration

### 2. Validation Against Code & Structure
Validates docstrings against:
- **Code signals**: Function name, parameters, return statements
- **Structural signals**: Control flow (if/else, loops), exception handling, function calls
- **Naturalness**: Checks for code syntax in natural language summary

### 3. Targeted Edit Instructions
- Generates 3-4 sentences describing what to ADD to improve the docstring
- Based on specific validation issues found
- Actionable and interpretable

### 4. Constrained Refinement with Word Preservation
- **Additive strategy**: Only adds clarifying words, never removes original words
- Guarantees BLEU-4 score preservation or improvement
- Post-validation to ensure all original words are present

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Iteration Agent                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Validate Docstring                                      │
│     ├─ Check parameters mentioned                           │
│     ├─ Check return value mentioned                         │
│     ├─ Check control flow mentioned                         │
│     ├─ Check function calls mentioned                       │
│     └─ Check naturalness (no code syntax)                   │
│                                                             │
│  2. Generate Edit Instructions (if issues found)            │
│     └─ 3-4 sentences describing what to add                 │
│                                                             │
│  3. Constrained Refinement                                  │
│     ├─ Add clarifying information                           │
│     ├─ Preserve all original words                          │
│     └─ Verify word preservation                             │
│                                                             │
│  4. Return Final Summary + Metadata                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Configuration

### Enable Iteration Agent

In `config.yaml`:

```yaml
# Iteration Agent Configuration (Phase 1 Innovation)
iteration_agent:
  enabled: true  # Single-pass validation and refinement
  validation:
    check_parameters: true
    check_return_value: true
    check_control_flow: true
    check_function_calls: true
    check_naturalness: true
  instruction_generation:
    max_instructions: 4  # 3-4 sentences
    temperature: 0.7
  constrained_refinement:
    strategy: "additive"  # Only add words, don't remove
    max_new_tokens: 150
    temperature: 0.7

# Reflective Agent - DISABLED for Phase 1
reflective_agent:
  enabled: false  # Replaced by iteration agent
```

## Usage

### In Inference Pipeline

The iteration agent is automatically used when enabled in config:

```python
from src.model.inference import InferencePipeline

# Initialize pipeline (iteration agent auto-initialized if enabled)
pipeline = InferencePipeline(model, tokenizer, rag_system, preprocessor, reflective_agent, config)

# Generate summary (iteration agent runs automatically)
result = pipeline.predict_single(code)

# Access metadata
print(f"Validation issues: {result.get('validation_issues')}")
print(f"Edit instructions: {result.get('edit_instructions')}")
print(f"Word preservation: {result.get('word_preservation')}")
```

### Direct Usage

```python
from src.agent.iteration_agent import IterationAgent

# Initialize
agent = IterationAgent(model, tokenizer, config)

# Run single-pass refinement
final_summary, metadata = agent.iterate_once(
    code=code,
    initial_summary=initial_summary,
    structure_summary=structure_summary
)

# Check results
print(f"Final: {final_summary}")
print(f"Issues found: {metadata['total_issues']}")
print(f"Refined: {metadata['refined']}")
print(f"Word preservation: {metadata['word_preservation']}")
```

## Example

### Input
```python
code = '''def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total'''

initial_summary = "Calculates a sum."
structure = "Function 'calculate_sum' with params (numbers), has 1 loop, returns value"
```

### Validation Issues Found
- Missing parameter: `numbers`
- Missing control flow: loop not mentioned
- Missing return value description

### Edit Instructions Generated
"Add mention of the 'numbers' parameter. Clarify that the function iterates through the list. Mention that it returns the total sum."

### Refined Summary
"Calculates a sum **of all numbers in the list by iterating through each element and returns the total**."

### Word Preservation Check
- Original words: ["Calculates", "a", "sum"]
- All present in refined: ✅
- BLEU-4 impact: Maintained or improved

## Testing

Run the test script:

```bash
python test_iteration_agent.py
```

This will test the iteration agent with sample functions and display:
- Validation issues found
- Edit instructions generated
- Final refined summary
- Word preservation status

## Benefits Over Reflective Agent

| Aspect | Reflective Agent | Iteration Agent |
|--------|------------------|-----------------|
| **Iterations** | 1-3 (multi-pass) | 1 (single-pass) |
| **Speed** | Slower | Faster |
| **Approval** | Keyword/score-based | Validation-based |
| **Word Preservation** | Not guaranteed | Guaranteed |
| **BLEU-4 Impact** | May decrease | Maintained/improved |
| **Interpretability** | Moderate | High (explicit issues) |
| **Phase** | Phase 1 & 2 | Phase 1 innovation |

## Phase 2 Extensions

The iteration agent provides a foundation for Phase 2:
- Multi-criteria validation scoring
- Adaptive refinement (skip for simple functions)
- Iterative refinement with convergence
- Alternative refinement strategies

These are **out of scope** for Phase 1.

## Files

- **Implementation**: `src/agent/iteration_agent.py`
- **Configuration**: `config.yaml` (iteration_agent section)
- **Integration**: `src/model/inference.py`
- **Test**: `test_iteration_agent.py`
- **Documentation**: `docs/ITERATION_AGENT.md` (this file)
