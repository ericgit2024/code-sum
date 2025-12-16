# Entity Refinement Agent - Issues and Fixes

## Issues Identified from User's Output

### **Issue 1: Entity Extraction Too Strict** ✅ FIXED

**Problem:**
The original entity extractor only found entities with very specific patterns like:
- "parameter param_name"
- "`param_name`"
- "calls function_name()"

But generated summaries use natural language like:
> "The function takes in a **price** and a **discount percentage**"

This didn't match any patterns, so entities weren't detected!

**Result:**
- Hallucination Score: 1.000 (100% hallucination!)
- Precision: 0.000 (no correct entities found)
- Recall: 0.000 (no required entities mentioned)

**Fix Applied:**
Enhanced `extract_from_docstring()` in `entity_extractor.py` to:
1. Extract ALL identifier words from the docstring
2. Filter out common English words (the, a, and, is, etc.)
3. Treat remaining words as potential entities
4. Add natural language patterns like "takes in a param_name"

**Result After Fix:**
- ✅ "price" detected in parameters
- ✅ "discount" detected in parameters
- ✅ Natural language docstrings now work!

---

### **Issue 2: Function Name Marked as Hallucinated** ⚠️ NEEDS INVESTIGATION

**Problem:**
The function name `calculate_discount` appears in the code but is marked as "hallucinated" in the docstring.

**Possible Causes:**
1. The docstring might not mention the function name explicitly
2. The entity extractor might not be looking for the function name in docstrings
3. The verifier might be comparing incorrectly

**Investigation Needed:**
- Check if docstring actually mentions "calculate_discount"
- Verify that function name extraction is working
- Ensure verifier compares function names correctly

---

### **Issue 3: Approval with Recall = 0.000** ⚠️ CRITICAL BUG

**Problem:**
In iteration 3, the agent approved the summary with:
- Hallucination Score: 0.000 ✓ (good)
- Precision: 1.000 ✓ (good)
- **Recall: 0.000** ✗ (BAD - means NO required entities mentioned!)
- F1: 0.000 ✗ (BAD)

**Why This is Wrong:**
Recall = 0.000 means the summary mentions ZERO of the required entities (parameters, function calls, etc.). This should FAIL verification, not pass!

**Root Cause:**
The hallucination threshold only checks `hallucination_score`, not recall or F1.

**Fix Needed:**
Update `EntityVerificationResult.passes_threshold` to also require:
- Minimum recall (e.g., 0.5 or 50%)
- Minimum F1 score (e.g., 0.4)

---

### **Issue 4: Incomplete Final Summary** ⚠️ TEXT CLEANING BUG

**Problem:**
Final summary ends with incomplete sentence:
> "...If the item is a member, the function also applies a 5% discount to the final price. **If.**"

**Cause:**
The `_clean_summary()` or `_enforce_max_sentences()` method is cutting off text incorrectly.

**Fix Needed:**
Review text cleaning logic in `EntityRefinementAgent._clean_summary()`

---

## Fixes Applied

### ✅ **Fix 1: Enhanced Entity Extraction**

**File:** `src/verification/entity_extractor.py`

**Changes:**
```python
# Added word-based extraction
identifier_pattern = r'\b([a-z_][a-z0-9_]*)\b'
all_words = set(re.findall(identifier_pattern, docstring_lower))

# Filter common English words
common_words = {'the', 'a', 'and', 'or', 'if', ...}
potential_entities = all_words - common_words

# Add all potential entities as parameters/variables
mentioned_parameters.update(potential_entities)
mentioned_variables.update(potential_entities)
```

**Impact:**
- Natural language docstrings now work
- Entities detected even without specific keywords
- Much more flexible and robust

---

## Fixes Still Needed

### ⚠️ **Fix 2: Add Recall/F1 Requirements to Verification**

**File:** `src/verification/entity_verifier.py`

**Current Logic:**
```python
passes_threshold = hallucination_score <= threshold
```

**Should Be:**
```python
passes_threshold = (
    hallucination_score <= threshold AND
    recall >= min_recall AND  # e.g., 0.5
    f1_score >= min_f1        # e.g., 0.4
)
```

**Why:**
- Prevents approving summaries that mention nothing (recall = 0)
- Ensures summaries are both accurate (low hallucination) AND complete (high recall)

---

### ⚠️ **Fix 3: Improve Text Cleaning**

**File:** `src/agent/entity_refinement_agent.py`

**Issue:**
The `_enforce_max_sentences()` method splits on periods, which can break on:
- Abbreviations (e.g., "Dr.", "etc.")
- Decimal numbers (e.g., "0.95")
- Incomplete sentences

**Suggested Fix:**
Use a more robust sentence splitter or check for incomplete sentences before truncating.

---

## Configuration Recommendations

### **Update `config.yaml`**

Add minimum thresholds for recall and F1:

```yaml
entity_verification:
  enabled: true
  
  # Thresholds
  hallucination_threshold: 0.30  # Max hallucination score
  min_recall: 0.50              # Min 50% of required entities mentioned
  min_f1_score: 0.40            # Min F1 score for balance
  
  # ... rest of config ...
```

---

## Testing Recommendations

### **Test Cases to Add:**

1. **Natural Language Docstring**
   - Verify entities extracted from "takes in a price" format
   
2. **Function Name Verification**
   - Ensure function name is correctly identified
   - Check if mentioning function name in docstring is required
   
3. **Recall Threshold**
   - Test that summaries with recall < 0.5 are rejected
   - Verify feedback asks to add missing entities
   
4. **Text Cleaning**
   - Test that final summaries don't have incomplete sentences
   - Verify decimal numbers don't break sentence splitting

---

## Summary

### **Fixed:**
✅ Entity extraction now works with natural language docstrings

### **Still Needs Fixing:**
⚠️ Add recall and F1 requirements to verification
⚠️ Fix text cleaning to prevent incomplete sentences
⚠️ Investigate function name hallucination issue

### **Impact:**
The entity refinement agent is now much more robust at detecting entities in natural language, but still needs improvements to prevent approving incomplete summaries and to fix text cleaning issues.
