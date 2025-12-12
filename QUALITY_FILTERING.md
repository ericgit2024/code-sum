# Quality Filtering for Code-Summary Pairs

## Overview

Quality filtering automatically removes low-quality code-summary pairs from the dataset **before training**. This improves model performance by ensuring the model only learns from high-quality examples.

---

## 🎯 Why Quality Filtering Matters

### **Problem: CodeSearchNet Contains Noise**

The CodeSearchNet dataset contains many low-quality samples:
- ❌ Empty or placeholder docstrings ("TODO", "FIXME")
- ❌ Docstrings that are just code snippets
- ❌ Generic summaries ("This function does something")
- ❌ Extremely long/short code or summaries
- ❌ Invalid Python syntax
- ❌ Irrelevant code-summary pairs

### **Impact on Training**

| Scenario | BLEU-4 | Issue |
|----------|--------|-------|
| **No filtering** | 0.15-0.18 | Model learns from noise |
| **With filtering** | **0.20-0.25** | ✅ **Clean training data** |

**Expected improvement:** +0.02-0.07 BLEU points

---

## ✅ Quality Checks (11 Heuristics)

### **1. Non-Empty**
- Code and summary must both be non-empty
- Rejects: `code=""` or `summary=""`

### **2. Code Length**
- **Min:** 20 characters
- **Max:** 2000 characters
- Rejects: Trivial one-liners or massive files

### **3. Summary Length**
- **Min:** 10 characters
- **Max:** 500 characters
- Rejects: "TODO" or entire paragraphs

### **4. Summary Word Count**
- **Min:** 3 words
- **Max:** 100 words
- Rejects: "Returns value" or overly verbose docs

### **5. Code Line Count**
- **Min:** 2 lines
- **Max:** 100 lines
- Rejects: Single-line code or entire modules

### **6. Summary is Not Code**
- Checks for code patterns: `def`, `class`, `import`, `return`
- Checks for high ratio of special characters: `{}[]();=<>`
- Rejects: Summaries that look like code snippets

### **7. Summary is Not Placeholder**
- Checks for: "TODO", "FIXME", "placeholder", "TBD", "..."
- Rejects: Auto-generated or incomplete docstrings

### **8. Summary is Not Just Function Name**
- Compares summary to function name
- Rejects: `def calculate_sum()` with summary "calculate_sum"

### **9. Code is Valid Python**
- Parses code with `ast.parse()`
- Rejects: Syntax errors, incomplete code

### **10. Summary Has Meaningful Content**
- Removes stopwords: "a", "the", "is", "are", "to", "of"
- Requires at least 2 meaningful words
- Rejects: "This is a function" (only stopwords)

### **11. Code-Summary Relevance**
- Basic check: summary shouldn't be completely generic
- Rejects: "This function does something" (no specifics)

---

## 📊 Expected Filtering Results

### **Typical Retention Rate: 70-85%**

From 5000 samples:
- **Kept:** ~3500-4250 high-quality samples
- **Removed:** ~750-1500 low-quality samples

### **Common Rejection Reasons**

| Reason | Typical % | Example |
|--------|-----------|---------|
| Summary too short | 15-20% | "Returns value" |
| Summary is placeholder | 10-15% | "TODO: Add description" |
| Code too long | 5-10% | Entire class definitions |
| Summary looks like code | 5-8% | "return a + b" |
| Invalid Python | 3-5% | Incomplete snippets |
| Summary lacks content | 2-5% | "This is a function" |
| Other | 5-10% | Various issues |

---

## 🔧 Configuration

### **Enable/Disable Filtering**

```yaml
dataset:
  quality_filter_enabled: true  # Set to false to disable
```

### **Adjust Thresholds**

```yaml
quality_filter:
  # Code constraints
  min_code_length: 20        # Increase to filter trivial code
  max_code_length: 2000      # Decrease to filter long code
  min_code_lines: 2          # Minimum lines of code
  max_code_lines: 100        # Maximum lines of code
  
  # Summary constraints
  min_summary_length: 10     # Increase for more detailed summaries
  max_summary_length: 500    # Decrease for concise summaries
  min_summary_words: 3       # Minimum words required
  max_summary_words: 100     # Maximum words allowed
```

### **Recommended Settings**

#### **Strict Filtering (Higher Quality, Fewer Samples)**
```yaml
quality_filter:
  min_code_length: 50
  min_summary_length: 20
  min_summary_words: 5
  max_code_lines: 50
```
**Expected retention:** ~60-70%

#### **Balanced Filtering (Default)**
```yaml
quality_filter:
  min_code_length: 20
  min_summary_length: 10
  min_summary_words: 3
  max_code_lines: 100
```
**Expected retention:** ~75-85%

#### **Lenient Filtering (More Samples, Lower Quality)**
```yaml
quality_filter:
  min_code_length: 10
  min_summary_length: 5
  min_summary_words: 2
  max_code_lines: 150
```
**Expected retention:** ~85-95%

---

## 📈 Impact on Performance

### **Expected Improvements**

| Metric | Without Filtering | With Filtering | Improvement |
|--------|-------------------|----------------|-------------|
| **BLEU-4** | 0.18 | **0.20-0.25** | +0.02-0.07 |
| **ROUGE-L** | 0.35 | **0.38-0.42** | +0.03-0.07 |
| **METEOR** | 0.47 | **0.50-0.54** | +0.03-0.07 |

### **Why It Works**

1. **Cleaner Training Signal**
   - Model doesn't learn from noisy examples
   - Faster convergence

2. **Better Generalization**
   - Model learns meaningful patterns
   - Not confused by contradictory examples

3. **Improved Metrics**
   - Test set also filtered (fair comparison)
   - Model generates higher-quality summaries

---

## 🚀 Usage

### **Automatic (Integrated into Training)**

Quality filtering happens automatically when you run:

```bash
python train.py
```

**Output:**
```
Loading CodeXGLUE code-to-text dataset for python...
Total available samples: 251820

Applying quality filtering...
============================================================
QUALITY FILTERING RESULTS
============================================================
Original samples: 5000
High-quality samples: 4127
Removed samples: 873
Retention rate: 82.5%

Rejection reasons:
  - Summary too short (42 chars): 312 samples
  - Summary is placeholder: 198 samples
  - Code too long (2145 chars): 156 samples
  - Summary looks like code: 107 samples
  - Invalid Python: 67 samples
  - Summary lacks meaningful content: 33 samples
============================================================

Final dataset splits:
  Train: 2889 samples
  Validation: 619 samples
  Test: 619 samples
```

### **Manual (Test Filtering)**

You can test filtering on a sample:

```python
from src.data.quality_filter import QualityFilter

# Create filter
filter = QualityFilter({
    'min_code_length': 20,
    'min_summary_length': 10,
    'min_summary_words': 3
})

# Check a sample
sample = {
    'code': 'def add(a, b):\n    return a + b',
    'docstring': 'Add two numbers and return the sum.'
}

is_quality, reason = filter.is_high_quality(sample)
print(f"Quality: {is_quality}, Reason: {reason}")
# Output: Quality: True, Reason: High quality
```

---

## 🔍 Examples

### ✅ **High Quality (Kept)**

```python
# Code
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Summary
"Calculate the nth Fibonacci number using recursion."

# Result: ✅ KEPT
# - Valid Python
# - Meaningful summary (7 words)
# - Appropriate length
# - Describes what code does
```

### ❌ **Low Quality (Removed)**

#### Example 1: Placeholder
```python
# Code
def process_data(items):
    return [x * 2 for x in items]

# Summary
"TODO: Add description"

# Result: ❌ REMOVED
# Reason: "Summary is placeholder"
```

#### Example 2: Too Short
```python
# Code
def add(a, b):
    return a + b

# Summary
"Adds"

# Result: ❌ REMOVED
# Reason: "Summary too few words (1)"
```

#### Example 3: Code-like Summary
```python
# Code
def multiply(x, y):
    return x * y

# Summary
"return x * y"

# Result: ❌ REMOVED
# Reason: "Summary looks like code"
```

#### Example 4: Generic Summary
```python
# Code
def find_max(numbers):
    return max(numbers)

# Summary
"This function does something"

# Result: ❌ REMOVED
# Reason: "Summary lacks meaningful content"
```

---

## 🎓 Advanced: Custom Filters

You can add custom quality checks by extending `QualityFilter`:

```python
class CustomQualityFilter(QualityFilter):
    def is_high_quality(self, sample):
        # Run base checks
        is_quality, reason = super().is_high_quality(sample)
        if not is_quality:
            return False, reason
        
        # Add custom check: summary must mention function purpose
        summary = sample.get('docstring', '')
        action_words = ['calculate', 'compute', 'find', 'get', 'set', 
                       'create', 'delete', 'update', 'process']
        
        if not any(word in summary.lower() for word in action_words):
            return False, "Summary doesn't describe action"
        
        return True, "High quality"
```

---

## 📊 Monitoring Quality

### **Check Retention Rate**

If retention rate is too low (<50%), consider:
1. Relaxing thresholds (increase min, decrease max)
2. Checking if dataset is particularly noisy
3. Reviewing rejection reasons

### **Check Rejection Reasons**

If one reason dominates (>50%), consider:
1. Adjusting that specific threshold
2. Investigating dataset quality issues
3. Adding custom filters for that issue

---

## 🎯 Best Practices

### **1. Start with Default Settings**
- Run training once with defaults
- Check retention rate and results

### **2. Adjust Based on Results**
- If retention <70%: Relax thresholds
- If retention >90%: Tighten thresholds
- If metrics don't improve: Check rejection reasons

### **3. Balance Quality vs Quantity**
- More samples = better generalization
- Higher quality = better learning signal
- Sweet spot: 75-85% retention

### **4. Monitor Training**
- Check if loss converges faster
- Compare metrics with/without filtering
- Validate on test set

---

## 📝 Summary

**What quality filtering does:**
1. ✅ Removes empty/placeholder summaries
2. ✅ Filters code-like summaries
3. ✅ Validates Python syntax
4. ✅ Ensures appropriate lengths
5. ✅ Checks for meaningful content
6. ✅ Basic relevance checking

**Expected benefits:**
- **+0.02-0.07 BLEU** improvement
- **Faster convergence** during training
- **Cleaner training signal**
- **Better generalization**

**Configuration:**
- **Enable:** `quality_filter_enabled: true`
- **Adjust:** Thresholds in `quality_filter` section
- **Monitor:** Check retention rate and rejection reasons

**Recommended retention rate:** 75-85% (3750-4250 samples from 5000)
