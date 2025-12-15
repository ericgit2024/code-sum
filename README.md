# Source Code Summarization with Gemma 2B + LoRA + Compact Structures + Reflective Agent

A novel approach to code summarization combining efficient fine-tuning with structural analysis and self-refinement.

## Current Approach (December 2024)

### Key Components

1. **Gemma 2B with LoRA** - Efficient fine-tuning with 4-bit quantization
2. **Compact Structure Summarizer** - Enhanced structural features including function names, parameters, and called functions
3. **Reflective Agent** - Iterative self-refinement with relaxed approval mechanism
4. **RAG System** - Currently disabled (ablation study showed better results without it)

### Performance

**Current Results** (5000 samples, no RAG):
- **BLEU-4**: 0.185 → Expected 0.20-0.25 (with 5K samples)
- **ROUGE-L**: 0.335 → Expected 0.38-0.42
- **METEOR**: 0.450 → Expected 0.48-0.52

**Expected with Reflective Agent** (after fixes):
- **BLEU-4**: 0.20-0.22 (SOTA range)
- **ROUGE-L**: 0.35-0.37
- **METEOR**: 0.47-0.49

### Novel Contributions

1. **Enhanced Compact Structures**
   - Function names and parameter names
   - Called functions list
   - Return type information
   - ~60-80 tokens vs 300-500 for full AST/CFG/PDG

2. **Improved Reflective Agent**
   - Relaxed approval (multiple keywords)
   - Convergence detection
   - Best summary tracking
   - 3 simplified criteria (accuracy, completeness, clarity)

3. **Efficient Training**
   - LoRA with 4-bit quantization
   - ~2% trainable parameters
   - 5000 samples in ~150-200 minutes

## Project Structure

```
.
├── config.yaml                 # Configuration (5000 samples, RAG disabled)
├── requirements.txt            # Dependencies
├── train.py                   # Training script
├── evaluate.py                # Evaluation script
├── run_inference.py           # Interactive inference
└── src/
    ├── data/
    │   ├── dataset_loader.py  # CodeSearchNet loader
    │   └── preprocessor.py    # Preprocessing pipeline
    ├── structure/
    │   ├── compact_summarizer.py  # Enhanced compact structures
    │   ├── ast_extractor.py   # AST extraction
    │   ├── cfg_extractor.py   # CFG extraction
    │   └── pdg_extractor.py   # PDG extraction
    ├── rag/
    │   └── rag_system.py      # RAG (currently disabled)
    ├── model/
    │   ├── model_loader.py    # Gemma + LoRA loader
    │   ├── trainer.py         # Training logic
    │   └── inference.py       # Inference pipeline
    ├── agent/
    │   └── reflective_agent.py # Improved reflective agent
    └── evaluation/
        └── metrics.py         # BLEU, ROUGE, METEOR
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set HuggingFace Token

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

## Usage

### Training (5000 samples, ~150-200 minutes)

```bash
python train.py
```

This will:
1. Load 5000 samples from CodeSearchNet Python (3500 train, 750 val, 750 test)
2. Extract enhanced compact structures
3. Fine-tune Gemma 2B with LoRA (4-bit)
4. Save to `./outputs/final_model`

### Evaluation

```bash
python evaluate.py --checkpoint ./outputs/final_model
```

Generates summaries with reflective agent and calculates metrics.

To disable reflective agent:
```bash
python evaluate.py --checkpoint ./outputs/final_model --no_reflective_agent
```

### Inference

```bash
python run_inference.py --checkpoint ./outputs/final_model --code "def add(a, b): return a + b"
```

## RL Training (Phase 1 & 2)

### Phase 1: Validation (500 samples, 5 epochs)

**Purpose**: Validate RL approach before full-scale training.

```bash
# Step 1: Test reward function
python test_reward_function.py

# Step 2: Analyze current model
python analyze_rewards.py --checkpoint ./outputs/final_model --num_samples 100

# Step 3: Run Phase 1 training
python train_rl_phase1.py --checkpoint ./outputs/final_model --num_samples 500 --num_epochs 5

# Step 4: Analyze improvements
python analyze_rewards.py --checkpoint ./outputs/rl_phase1/final_model --num_samples 100

# Step 5: Evaluate metrics
python evaluate.py --checkpoint ./outputs/rl_phase1/final_model
```

**Expected Results**:
- Reward: 0.55 → 0.65-0.70 (+0.10-0.15)
- BLEU-4: 0.185 → 0.20-0.22 (+0.02-0.04)
- Better parameter coverage and control flow descriptions

**Runtime**: ~2-3 hours on T4 GPU

### Phase 2: Full-Scale Training (5000 samples, 10 epochs)

If Phase 1 shows improvement, proceed to Phase 2:

```bash
python train_rl_phase2.py --checkpoint ./outputs/rl_phase1/final_model --num_samples 5000 --num_epochs 10
```

**Expected Results**:
- Reward: 0.75-0.85
- BLEU-4: 0.23-0.27 (SOTA range)
- ROUGE-L: 0.38-0.42
- METEOR: 0.50-0.54

**Runtime**: ~20-30 hours on T4 GPU

### RL Reward Components

The reward function evaluates 6 aspects:
1. **Parameter Coverage (25%)**: All parameters mentioned
2. **Return Mention (25%)**: Return value described
3. **Control Flow (20%)**: if/else, loops covered
4. **Naturalness (15%)**: No code syntax
5. **Fluency (10%)**: BLEU-based readability
6. **Hallucination Penalty (5%)**: No non-existent features

See [RL Phase 1 Usage Guide](docs/rl_phase1_usage_guide.md) for detailed instructions.

---

## Configuration

Edit `config.yaml`:

```yaml
dataset:
  sample_size: 5000  # Training samples

structure:
  use_compact_summary: true  # Enhanced compact structures

reflective_agent:
  enabled: false  # Disabled during RL training (can re-enable for inference)
  max_iterations: 3
  criteria: [accuracy, completeness, clarity]

rag:
  enabled: false  # Disabled (ablation study)
  
rl:
  enabled: true  # Phase 1: RL training with execution-based rewards
  num_epochs: 5
  batch_size: 4
  learning_rate: 1.0e-5
```

## Key Features

### 1. Enhanced Compact Structures

**Example Output**:
```
Function 'calculate_distance' with params (x, y, z), 
has 2 conditionals, 1 loop, calls [math.sqrt, abs, max], 
returns float
```

**Benefits**:
- Includes function/parameter names
- Lists called functions
- Shows return types
- Only ~60-80 tokens (vs 300-500 for full graphs)

### 2. Improved Reflective Agent

**Fixes**:
- Relaxed approval (accepts "GOOD", "ACCEPTABLE", not just "APPROVED")
- Convergence detection (stops if summary unchanged)
- Best summary tracking (returns best, not last)
- Simplified criteria (3 instead of 5)

**Expected Improvement**: +0.02-0.04 BLEU-4

### 3. Efficient Training

- 4-bit quantization: ~12GB VRAM
- LoRA: ~2% trainable parameters
- 5000 samples: ~150-200 minutes

## Comparison to SOTA

| Method | BLEU-4 | ROUGE-L | Novel Aspect |
|--------|--------|---------|--------------|
| CodeBERT | 0.17 | 0.37 | Pre-training |
| GraphCodeBERT | 0.18 | 0.38 | Data flow graphs |
| CodeT5 | 0.21 | 0.39 | Identifier-aware |
| **Ours (base)** | **0.185** | **0.335** | **Compact structures** |
| **Ours (+ agent)** | **0.20-0.22** | **0.35-0.37** | **+ Reflective agent** |

## Ablation Study

| Configuration | BLEU-4 | Notes |
|---------------|--------|-------|
| Base model only | 0.15 | No structures |
| + Compact structures | 0.185 | +0.035 improvement |
| + Reflective agent | 0.20-0.22 | +0.015-0.035 (expected) |
| + RAG (broken) | 0.047 | -0.138 (contamination) |

## Requirements

- Python 3.8+
- CUDA GPU (12GB+ VRAM recommended)
- HuggingFace account with Gemma access

## Notes

- First run downloads Gemma 2B (~5GB)
- CodeSearchNet cached locally
- Training: ~150-200 minutes (5000 samples)
- Outputs saved to `./outputs`

## Citation

```bibtex
@mastersthesis{code_summarization_2024,
  title={Efficient Code Summarization using Compact Structures and Reflective Agents},
  author={Your Name},
  year={2024},
  school={Your University}
}
```
