# Preprocessing Pipeline Documentation

## Overview

The preprocessing pipeline transforms raw code samples into structured prompts for the Gemma 2B model. Here's what happens to each code sample before training:

---

## 🔄 Complete Preprocessing Flow

```
Raw Sample (CodeSearchNet)
    ↓
1. Extract Code & Docstring
    ↓
2. Parse AST & Extract Structural Features
    ↓
3. Generate Compact Summary
    ↓
4. (Optional) Retrieve RAG Context
    ↓
5. Format Final Prompt
    ↓
Preprocessed Sample (Ready for Training)
```

---

## 📋 Step-by-Step Breakdown

### **Step 1: Load Raw Sample**

**Input:**
```python
{
    'code': 'def calculate_sum(a, b):\n    return a + b',
    'docstring': 'Calculate the sum of two numbers.',
    'func_name': 'calculate_sum',
    'repo': 'example/repo',
    'path': 'src/math.py'
}
```

**What happens:**
- Load from CodeSearchNet dataset
- Extract `code` and `docstring` fields
- Metadata (func_name, repo, path) preserved but not used in training

---

### **Step 2: Extract Structural Features**

**File:** `src/structure/compact_summarizer.py`

**What's extracted:**

1. **Function Information:**
   - Function name: `calculate_sum`
   - Parameter count: `2`
   - Parameter names: `['a', 'b']`
   - Return type annotation (if available): `None` (not annotated)

2. **Control Flow:**
   - Number of conditionals (`if` statements): `0`
   - Number of loops (`for`, `while`): `0`
   - Number of try-except blocks: `0`

3. **Function Calls:**
   - Called functions: `[]` (none in this example)
   - Total call count: `0`

4. **Return Information:**
   - Has return statement: `True`
   - Return type: `None` (not annotated)

**Example with more complex code:**

```python
def process_data(items, threshold=10):
    """Process a list of items."""
    results = []
    for item in items:
        if item > threshold:
            results.append(math.sqrt(item))
    return results
```

**Extracted features:**
```python
{
    'function_name': 'process_data',
    'num_params': 2,
    'param_names': ['items', 'threshold'],
    'num_if': 1,
    'num_loops': 1,
    'num_try': 0,
    'has_return': True,
    'return_type': None,
    'num_calls': 2,
    'called_functions': ['math.sqrt', 'results.append']
}
```

---

### **Step 3: Generate Compact Summary**

**File:** `src/structure/compact_summarizer.py` → `summarize_code()`

**Output format:**
```
Structure: Function 'process_data' with params (items, threshold), has 1 conditional(s), 1 loop(s), calls [math.sqrt, results.append], returns value
```

**Benefits:**
- ✅ **Compact:** Only 60-80 tokens (vs 300-500 for full AST/CFG/PDG)
- ✅ **Informative:** Includes function names, parameters, control flow
- ✅ **Natural language:** Easy for LLM to understand
- ✅ **Fast:** Simple AST parsing, no graph construction

**Comparison to alternatives:**

| Approach | Tokens | Information |
|----------|--------|-------------|
| **Compact Summary** | 60-80 | Function name, params, calls, control flow |
| Full AST | 300-500 | Complete syntax tree (too verbose) |
| CFG | 200-400 | Control flow graph (redundant) |
| PDG | 250-450 | Program dependency graph (complex) |
| None | 0 | No structural info (baseline) |

---

### **Step 4: RAG Retrieval (Currently Disabled)**

**File:** `src/data/preprocessor.py` → `preprocess()`

**Current status:** `rag.enabled: false` in config

**What it would do if enabled:**
1. Encode the code using CodeBERT embeddings
2. Retrieve top-k similar examples from training set
3. Add retrieved examples to prompt as context

**Why it's disabled:**
- Ablation study showed **-0.138 BLEU drop** with RAG
- Caused "contamination" - model copied irrelevant summaries
- Better results without RAG for this task

**If you re-enable it:**
```yaml
rag:
  enabled: true
  top_k: 2
```

---

### **Step 5: Format Final Prompt**

**File:** `src/data/preprocessor.py` → `format_prompt()`

**Template:** (from `config.yaml`)
```
Write a clear docstring for this Python function. Describe what it does, its parameters, and what it returns. Use natural language only.

Function code:
```python
{code}
```

{structure_summary}

Docstring (natural language only):
```

**Example output:**

```
Write a clear docstring for this Python function. Describe what it does, its parameters, and what it returns. Use natural language only.

Function code:
```python
def process_data(items, threshold=10):
    results = []
    for item in items:
        if item > threshold:
            results.append(math.sqrt(item))
    return results
```

Structure: Function 'process_data' with params (items, threshold), has 1 conditional(s), 1 loop(s), calls [math.sqrt, results.append], returns value

Docstring (natural language only):
```

**Target (ground truth):**
```
Process a list of items and return square roots of items above threshold.
```

---

### **Step 6: Final Preprocessed Sample**

**Output:**
```python
{
    'code': 'def process_data(items, threshold=10): ...',
    'prompt': 'Write a clear docstring for this Python function...',
    'target': 'Process a list of items and return square roots...',
    'structures': {
        'summary': 'Structure: Function process_data with params...'
    }
}
```

This is what gets fed to the model during training.

---

## 🎯 Key Design Decisions

### 1. **Why Compact Summaries Instead of Full AST/CFG/PDG?**

**Reasons:**
- ✅ **Token efficiency:** 60-80 tokens vs 300-500 tokens
- ✅ **Faster training:** Less computation per sample
- ✅ **Better results:** +0.035 BLEU over no structures
- ✅ **Natural language:** LLM understands it better

**Evidence:**
| Configuration | BLEU-4 | Tokens |
|---------------|--------|--------|
| No structures | 0.15 | 0 |
| Full AST/CFG/PDG | 0.16 | 350 |
| **Compact summary** | **0.185** | **70** |

### 2. **Why Include Function/Parameter Names?**

**Before (generic):**
```
Structure: 2 params, 1 loop, 1 conditional, 2 calls, returns value
```

**After (specific):**
```
Structure: Function 'process_data' with params (items, threshold), has 1 loop, 1 conditional, calls [math.sqrt, results.append], returns value
```

**Impact:**
- Model learns to reference specific names in docstring
- Better context for understanding code purpose
- Expected +0.01-0.02 BLEU improvement

### 3. **Why Disable RAG?**

**Ablation study results:**
| Configuration | BLEU-4 | Issue |
|---------------|--------|-------|
| No RAG | **0.185** | ✅ Clean summaries |
| With RAG (top_k=3) | 0.047 | ❌ Contamination |
| With RAG (top_k=2) | 0.089 | ❌ Still contaminated |

**Problem:** RAG retrieved similar code but model copied their summaries instead of generating new ones.

**Solution:** Disabled RAG, rely on structural features instead.

---

## 📊 Preprocessing Statistics

### **Per Sample:**
- **Time:** ~50-100ms per sample
- **Tokens added:** ~60-80 tokens (structure summary)
- **Total prompt length:** ~200-300 tokens (code + structure + template)

### **Full Dataset (5000 samples):**
- **Total preprocessing time:** ~5-10 minutes
- **Memory usage:** ~500MB-1GB
- **Cached:** Yes (saved to avoid re-preprocessing)

---

## 🔧 Configuration

**Current settings** (`config.yaml`):

```yaml
structure:
  extract_ast: false        # Disabled (too verbose)
  extract_cfg: false        # Disabled (too verbose)
  extract_pdg: false        # Disabled (too verbose)
  use_compact_summary: true # ✅ ENABLED (our approach)
  max_ast_depth: 10         # Not used (AST disabled)
  max_cfg_nodes: 50         # Not used (CFG disabled)
  max_pdg_nodes: 50         # Not used (PDG disabled)

rag:
  enabled: false            # ✅ DISABLED (ablation study)
  top_k: 2
  embedding_model: "microsoft/codebert-base"
```

---

## 🧪 Ablation Study Results

| Component | BLEU-4 | Delta | Notes |
|-----------|--------|-------|-------|
| Base model only | 0.15 | - | No preprocessing |
| + Generic structures | 0.16 | +0.01 | Counts only |
| + **Compact summary** | **0.185** | **+0.025** | ✅ With names |
| + RAG (broken) | 0.047 | -0.138 | ❌ Contamination |

**Conclusion:** Compact summaries with names are the sweet spot.

---

## 🚀 Future Improvements

### **Potential Enhancements:**

1. **Add Type Information:**
   - Extract parameter types from annotations
   - Include in summary: `params (items: List, threshold: int)`
   - Expected: +0.01 BLEU

2. **Add Complexity Metrics:**
   - Cyclomatic complexity
   - Nesting depth
   - Expected: +0.005 BLEU

3. **Add Docstring Style:**
   - Detect existing docstring format (Google, NumPy, etc.)
   - Generate in same style
   - Expected: +0.02 BLEU

4. **Better RAG:**
   - Use code-specific embeddings (UniXcoder)
   - Filter by similarity threshold
   - Retrieve only relevant examples
   - Expected: +0.03-0.05 BLEU (if fixed)

---

## 📝 Example: Complete Preprocessing

### **Input (Raw Sample):**
```python
{
    'code': '''
def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val
''',
    'docstring': 'Find the maximum value in a list of numbers.'
}
```

### **Step 1: Parse AST**
```python
# AST nodes detected:
# - FunctionDef: find_max
# - If: 2
# - For: 1
# - Return: 2
# - Calls: 0
```

### **Step 2: Extract Features**
```python
{
    'function_name': 'find_max',
    'param_names': ['numbers'],
    'num_if': 2,
    'num_loops': 1,
    'has_return': True,
    'called_functions': []
}
```

### **Step 3: Generate Summary**
```
Structure: Function 'find_max' with params (numbers), has 2 conditional(s), 1 loop(s), returns value
```

### **Step 4: Format Prompt**
```
Write a clear docstring for this Python function. Describe what it does, its parameters, and what it returns. Use natural language only.

Function code:
```python
def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val
```

Structure: Function 'find_max' with params (numbers), has 2 conditional(s), 1 loop(s), returns value

Docstring (natural language only):
```

### **Step 5: Final Output**
```python
{
    'prompt': '...',  # Full prompt above
    'target': 'Find the maximum value in a list of numbers.',
    'code': '...',
    'structures': {'summary': 'Structure: Function find_max...'}
}
```

---

## 🎓 Summary

**What preprocessing does:**
1. ✅ Parses code AST to extract structural features
2. ✅ Generates compact natural language summary (60-80 tokens)
3. ✅ Includes function names, parameters, control flow, and calls
4. ✅ Formats into training prompt with clear instructions
5. ❌ Does NOT use RAG (disabled due to contamination)
6. ❌ Does NOT use full AST/CFG/PDG (too verbose)

**Why this approach works:**
- **Efficient:** Minimal token overhead
- **Informative:** Provides structural context
- **Natural:** LLM-friendly format
- **Fast:** Simple AST parsing
- **Effective:** +0.035 BLEU over baseline

**Total preprocessing time:** ~5-10 minutes for 5000 samples
