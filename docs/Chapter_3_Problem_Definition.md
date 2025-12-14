# Chapter 3: Problem Definition

This chapter formally defines the problem of efficient code summarization, including the scope, assumptions, constraints, and key challenges that this thesis addresses.

---

## 3.1 Formal Problem Statement

### 3.1.1 Conceptual Formulation

**Problem:** Given a source code function $f$ written in Python, generate a concise natural language summary $s$ that accurately describes:
1. The function's purpose and behavior
2. Input parameters and their roles
3. Return value and its meaning
4. Key algorithmic logic (conditionals, loops, exceptions)

**Constraints:**
- The summary $s$ must be human-readable natural language (not code)
- Length: 10-100 words (concise but complete)
- Generated using limited computational resources (consumer-grade GPU, ~12GB VRAM)
- Inference time: <5 seconds per function
- Quality must be competitive with state-of-the-art methods

**Input:** 
- Source code function $f \in \mathcal{F}$ where $\mathcal{F}$ is the set of all valid Python functions
- Optional: Structural metadata $m$ extracted from $f$ (function signature, control flow statistics, called functions)

**Output:**
- Natural language summary $s \in \mathcal{S}$ where $\mathcal{S}$ is the set of all valid English summaries

**Objective:** Learn a mapping $\phi: \mathcal{F} \times \mathcal{M} \rightarrow \mathcal{S}$ that maximizes:
- **Accuracy:** Summary correctly describes what the code does
- **Completeness:** Summary covers all important aspects (parameters, logic, return)
- **Naturalness:** Summary reads like human-written documentation
- **Conciseness:** Summary is clear and to-the-point without verbosity

### 3.1.2 Mathematical Formulation

**Optimization Objective:**

Given a dataset $\mathcal{D} = \{(f_i, s_i^*)\}_{i=1}^{N}$ of code-summary pairs, we aim to learn model parameters $\theta$ that minimize:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(s_i^* \mid f_i, m_i)$$

where:
- $f_i$ = source code function
- $s_i^*$ = ground truth summary
- $m_i$ = compact structural metadata extracted from $f_i$
- $P_\theta(s \mid f, m)$ = probability of generating summary $s$ given code $f$ and metadata $m$ under model parameters $\theta$

**Structural Metadata Extraction:**

The compact structure $m$ is defined as:

$$m = \text{Extract}(f) = \{n, p, r, c_{if}, c_{loop}, c_{try}, \mathcal{C}\}$$

where:
- $n$ = function name
- $p = \{p_1, p_2, \ldots, p_k\}$ = parameter names
- $r$ = return type (if annotated)
- $c_{if}$ = count of conditional statements
- $c_{loop}$ = count of loops
- $c_{try}$ = count of exception handlers
- $\mathcal{C} = \{c_1, c_2, \ldots, c_j\}$ = set of called functions

**Constraint:** $|m| \leq 80$ tokens (vs $|AST(f)| \approx 300-500$ tokens)

**Reflective Refinement (Phase 1):**

For a given code $f$ and initial summary $s_0$, the reflective agent iteratively refines:

$$s_{t+1} = \begin{cases}
s_t & \text{if } \text{Approve}(s_t, f) = \text{True} \\
\text{Refine}(s_t, \text{Critique}(s_t, f)) & \text{otherwise}
\end{cases}$$

for $t = 0, 1, \ldots, T-1$ where $T \leq 3$ (max iterations)

**Approval Function (Phase 1 - Keyword-based):**

$$\text{Approve}(s, f) = \begin{cases}
\text{True} & \text{if } \exists k \in \mathcal{K}_{approve}: k \in \text{Critique}(s, f) \\
\text{False} & \text{if } \exists k \in \mathcal{K}_{reject}: k \in \text{Critique}(s, f) \\
\text{False} & \text{otherwise}
\end{cases}$$

where:
- $\mathcal{K}_{approve}$ = {"APPROVED", "GOOD", "ACCEPTABLE", ...}
- $\mathcal{K}_{reject}$ = {"NOT APPROVED", "NEEDS IMPROVEMENT", ...}
- Rejection keywords override approval keywords

**Reflective Refinement (Phase 2 - Score-based):**

$$\text{Approve}(s, f) = \begin{cases}
\text{True} & \text{if } S(s, f) \geq \tau_{approve} \\
\text{False} & \text{otherwise}
\end{cases}$$

where the weighted score $S$ is:

$$S(s, f) = w_1 \cdot A_{acc}(s, f) + w_2 \cdot A_{comp}(s, f) + w_3 \cdot A_{nat}(s, f) + w_4 \cdot A_{conc}(s, f)$$

with:
- $A_{acc}, A_{comp}, A_{nat}, A_{conc} \in [0, 1]$ = scores for accuracy, completeness, naturalness, conciseness
- $w_1 = 0.35, w_2 = 0.30, w_3 = 0.20, w_4 = 0.15$ (weights sum to 1.0)
- $\tau_{approve} = 0.75$ (approval threshold)

**Early Stopping Conditions:**

Stop refinement if:
1. $S(s_t, f) \geq 0.90$ (excellent quality)
2. $S(s_t, f) - S(s_{t-1}, f) < 0.05$ (minimal improvement)
3. $s_t = s_{t-1}$ (convergence)
4. $t = T$ (max iterations)

**Adaptive Iterations (Phase 2):**

$$T = \begin{cases}
1 & \text{if } CC(f) \leq 3 \text{ (simple)} \\
2 & \text{if } 3 < CC(f) \leq 8 \text{ (moderate)} \\
3 & \text{if } CC(f) > 8 \text{ (complex)}
\end{cases}$$

where $CC(f)$ is the cyclomatic complexity:

$$CC(f) = 1 + \sum_{v \in V} \mathbb{1}[v \in \{\text{If, While, For, ExceptHandler}\}] + \sum_{b \in B} (|b.values| - 1)$$

with $V$ = all AST nodes, $B$ = all BoolOp nodes

**Evaluation Metrics:**

Model quality is measured using:

$$\text{BLEU-4}(s, s^*) = BP \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

$$\text{ROUGE-L}(s, s^*) = \frac{(1 + \beta^2) \cdot R_{lcs} \cdot P_{lcs}}{\beta^2 \cdot R_{lcs} + P_{lcs}}$$

$$\text{METEOR}(s, s^*) = F_{mean} \cdot (1 - \text{Penalty})$$

where $s$ = generated summary, $s^*$ = reference summary

---

## 3.2 Scope and Assumptions

### 3.2.1 Scope

**In Scope:**
1. **Single-function summarization:** Generating summaries for individual Python functions
2. **Structural analysis:** Extracting lightweight metadata (function signatures, control flow, calls)
3. **Fine-tuned LLM:** Using Gemma 2B with LoRA for efficient training
4. **Reflective refinement:** Iterative critique and improvement of summaries
5. **Consumer hardware:** Optimization for GPUs with ~12GB VRAM
6. **Benchmark evaluation:** BLEU, ROUGE, METEOR on CodeSearchNet dataset

**Out of Scope:**
1. **Multi-file analysis:** Cross-file dependencies and repository-level understanding (future work)
2. **Non-Python languages:** C++, Java, JavaScript, etc. (future work)
3. **Runtime behavior:** Dynamic analysis, profiling, execution traces
4. **Code generation:** Reverse task (summary → code)
5. **Bug detection:** Identifying errors or vulnerabilities
6. **Large-scale deployment:** Production systems, API services

### 3.2.2 Assumptions

1. **Code Quality:** Input code is syntactically valid Python (can be parsed by `ast` module)
2. **Function Length:** Functions are 2-100 lines (manageable complexity)
3. **Documentation Style:** Ground truth summaries follow natural language conventions (not code examples)
4. **Dataset Quality:** CodeSearchNet provides representative Python functions with reasonable summaries
5. **Hardware Availability:** Access to GPU with at least 12GB VRAM for training/inference
6. **Model Access:** Ability to use Gemma 2B model (requires HuggingFace authentication)
7. **Evaluation Metrics:** BLEU, ROUGE, METEOR are valid proxies for summary quality
8. **Iteration Limit:** 3 iterations are sufficient for quality improvement (diminishing returns beyond)

### 3.2.3 Constraints

**Computational Constraints:**
- **Memory:** ≤12GB VRAM (consumer GPU: RTX 3090, RTX 4090)
- **Training Time:** ≤4 hours for 5,000 samples
- **Inference Time:** ≤5 seconds per function (including reflective iterations)
- **Model Size:** ≤5GB on disk (quantized)

**Quality Constraints:**
- **BLEU-4:** ≥0.18 (competitive with baselines)
- **Summary Length:** 10-100 words (concise but complete)
- **Naturalness:** No code syntax in summaries (e.g., no `def`, `if`, `for`)
- **Accuracy:** No hallucinations or incorrect descriptions

**Dataset Constraints:**
- **Size:** 5,000 samples (limited by training time)
- **Language:** Python only
- **Domain:** General-purpose code (not domain-specific)
- **Quality:** Filtered to remove malformed or trivial examples

**Architectural Constraints:**
- **Base Model:** Gemma 2B (cannot use larger models due to memory)
- **Fine-tuning:** LoRA only (full fine-tuning too expensive)
- **Quantization:** 4-bit (8-bit or full precision exceeds memory)
- **Structural Analysis:** Compact summarizer only (full AST/CFG/PDG too verbose)

---

## 3.3 Challenges

This section outlines the major research barriers and technical difficulties addressed in this thesis.

### 3.3.1 Computational Efficiency vs. Quality Trade-off

**Challenge:** State-of-the-art code summarization models (e.g., CodeT5, GraphCodeBERT) require:
- Large models (220M-770M parameters)
- Full fine-tuning (100% parameters updated)
- High-end GPUs (32GB+ VRAM)
- Long training times (8-12 hours)

**Impact:** Inaccessible to researchers/developers with consumer hardware

**Our Approach:**
- Use smaller model (Gemma 2B) with LoRA (~2% trainable parameters)
- 4-bit quantization (12GB VRAM)
- Maintain competitive quality (BLEU: 0.185-0.20 vs SOTA 0.21)

**Remaining Difficulty:** Balancing model capacity with memory constraints

---

### 3.3.2 Structural Information Overload

**Challenge:** Traditional approaches use full graph representations:
- **AST (Abstract Syntax Tree):** 300-500 tokens per function
- **CFG (Control Flow Graph):** 200-300 tokens
- **PDG (Program Dependence Graph):** 250-400 tokens

**Impact:** 
- Exceeds LLM context window limits
- Increases computational cost
- Introduces noise (irrelevant structural details)

**Our Approach:**
- Compact structure summarizer (60-80 tokens)
- Extract only high-level features (function name, params, control flow counts, key calls)
- 75-85% token reduction while retaining 95% of information value

**Remaining Difficulty:** Determining which structural features are most informative

---

### 3.3.3 Summary Quality Validation

**Challenge:** Generated summaries often suffer from:
- **Hallucinations:** Describing functionality that doesn't exist
- **Incompleteness:** Missing parameters, return values, or key logic
- **Code contamination:** Including code syntax instead of natural language
- **Verbosity:** Overly long or repetitive descriptions

**Impact:** Low BLEU/ROUGE/METEOR scores, poor user experience

**Our Approach:**
- Reflective agent that critiques and refines summaries
- Multi-criteria evaluation (accuracy, completeness, naturalness, conciseness)
- Best summary tracking (returns highest quality, not last)
- Convergence detection (stops when summary stops improving)

**Remaining Difficulty:** Designing effective critique prompts that guide refinement

---

### 3.3.4 Keyword-based Approval Limitations (Phase 1)

**Challenge:** Phase 1 uses keyword-based approval:
- Binary decision (approved/not approved)
- No quantitative quality assessment
- Cannot track improvement over iterations
- Sensitive to prompt phrasing

**Impact:** 
- May approve mediocre summaries
- May reject good summaries due to missing keywords
- No visibility into why summary was approved/rejected

**Our Approach (Phase 2):**
- Multi-criteria scoring (0.0-1.0 per criterion)
- Weighted aggregate score
- Early stopping on excellent scores (≥0.90)
- Minimal improvement detection (<0.05)

**Remaining Difficulty:** Ensuring LLM generates consistent numerical scores

---

### 3.3.5 Fixed Iteration Inefficiency (Phase 1)

**Challenge:** Phase 1 uses fixed 3 iterations for all functions:
- **Simple functions:** Over-processed (1 iteration sufficient)
- **Complex functions:** Under-processed (3 iterations may not be enough)
- **Evaluation bottleneck:** 3 iterations per function slows benchmarking

**Impact:** 
- Wasted computation on simple code
- Suboptimal quality on complex code
- Slow evaluation (3x longer than necessary)

**Our Approach (Phase 2):**
- Adaptive iterations based on cyclomatic complexity
- Simple (CC ≤ 3): 1 iteration
- Moderate (CC 4-8): 2 iterations
- Complex (CC > 8): 3 iterations

**Remaining Difficulty:** Accurately assessing code complexity in real-time

---

### 3.3.6 RAG Contamination Problem

**Challenge:** Initial RAG implementation retrieved similar code examples to augment context:
- Retrieved examples leaked into generated summaries
- Summaries described retrieved code, not target code
- Massive performance drop (BLEU: 0.185 → 0.047)

**Impact:** 
- RAG unusable for single-function summarization
- Wasted implementation effort

**Our Approach:**
- Disable RAG for Phase 1 (validated by ablation study)
- Focus on structural analysis instead
- Reserve RAG for repository-level analysis (future work)

**Remaining Difficulty:** Designing contamination-free retrieval for multi-file tasks

---

### 3.3.7 Evaluation Speed vs. Quality Trade-off

**Challenge:** Reflective agent improves quality but slows evaluation:
- 3 iterations per function
- Each iteration requires LLM generation (critique + refinement)
- Benchmarking 750 test samples takes hours

**Impact:** 
- Slow iteration on model improvements
- Difficult to run ablation studies
- Impractical for large-scale evaluation

**Our Approach:**
- Evaluation mode: 1 iteration instead of 3 (2-3x speedup)
- Greedy decoding option (30-50% faster generation)
- Token limits for critique/refinement (250/300 tokens)

**Remaining Difficulty:** Balancing evaluation speed with representative quality assessment

---

### 3.3.8 Generalization to Unseen Code Patterns

**Challenge:** Model trained on CodeSearchNet may not generalize to:
- Domain-specific code (ML, web, systems)
- Novel libraries or frameworks
- Uncommon coding patterns
- Different documentation styles

**Impact:** 
- Lower quality on out-of-distribution code
- Limited real-world applicability

**Our Approach:**
- Use diverse CodeSearchNet dataset (5,000 samples)
- Quality filtering to ensure variety
- Structural analysis provides domain-agnostic features

**Remaining Difficulty:** Ensuring robustness across all Python domains

---

## 3.4 Research Questions

Based on the challenges above, this thesis addresses the following research questions:

**RQ1:** Can a compact structural representation (60-80 tokens) achieve comparable performance to full graph representations (300-500 tokens) for code summarization?

**RQ2:** Does iterative refinement via a reflective agent improve summary quality compared to single-pass generation?

**RQ3:** Can multi-criteria scoring (Phase 2) outperform keyword-based approval (Phase 1) in terms of quality and efficiency?

**RQ4:** Does adaptive iteration strategy (based on code complexity) improve resource allocation compared to fixed iterations?

**RQ5:** Can a small model (Gemma 2B) with efficient fine-tuning (LoRA + 4-bit) achieve competitive results with larger SOTA models on consumer hardware?

---

## 3.5 Success Criteria

This thesis is considered successful if:

1. **Performance:** BLEU-4 ≥ 0.18, ROUGE-L ≥ 0.33, METEOR ≥ 0.45 (competitive with baselines)
2. **Efficiency:** Training time ≤ 4 hours, inference time ≤ 5 seconds per function
3. **Resource Constraints:** Memory usage ≤ 12GB VRAM (consumer GPU)
4. **Ablation Validation:** Compact structures improve over base model by ≥0.03 BLEU
5. **Reflective Agent Impact:** Reflective agent improves over single-pass by ≥0.015 BLEU
6. **Phase 2 Improvements:** Scoring + adaptive iterations improve over Phase 1 by ≥0.015 BLEU
7. **Generalization:** Consistent performance across different code complexity levels

---

## Summary

This chapter formally defined the problem of efficient code summarization with the following key points:

- **Problem:** Generate natural language summaries for Python functions using limited computational resources
- **Scope:** Single-function summarization on consumer hardware (12GB VRAM)
- **Constraints:** Memory, time, quality, and dataset size limitations
- **Challenges:** Efficiency vs. quality trade-off, structural information overload, summary validation, RAG contamination, evaluation speed
- **Approach:** Compact structures + fine-tuned Gemma 2B + reflective agent (keyword-based → score-based)
- **Success Criteria:** Competitive performance (BLEU ≥ 0.18) with efficient resource usage

The next chapter (Chapter 4: Proposed Solution) will detail the technical approach to address these challenges.
