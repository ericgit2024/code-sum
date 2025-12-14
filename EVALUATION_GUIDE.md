# Quick Evaluation Guide

## Semantic Similarity Metric (BERTScore)

BERTScore has been added to measure semantic similarity between generated and reference summaries. Unlike BLEU/ROUGE which rely on exact n-gram matches, BERTScore uses contextual embeddings to capture semantic meaning.

### Installation

```bash
pip install bert-score
```

### Usage

**Full evaluation (all test samples):**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-XXXX --no_reflective_agent
```

**Quick testing (20 samples only):**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-XXXX --no_reflective_agent --num_samples 20
```

**Quick testing with fast mode:**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-XXXX --no_reflective_agent --num_samples 20 --fast_mode
```

### Command Line Arguments

- `--checkpoint`: Path to model checkpoint (required)
- `--num_samples`: Limit number of test samples (e.g., 20 for quick testing)
- `--no_reflective_agent`: Disable reflective agent (Phase 1 baseline)
- `--fast_mode`: Enable fast mode (greedy decoding, reduced tokens)
- `--max_iterations`: Override max iterations for reflective agent
- `--output`: Output path for results (default: evaluation_results/results.json)

### Metrics Explained

**BLEU-4**: N-gram precision (0.0-1.0)
- Measures exact word/phrase overlap
- Higher = more similar to reference

**ROUGE-L**: Longest common subsequence (0.0-1.0)
- Measures sentence-level similarity
- Higher = better fluency

**METEOR**: Semantic similarity with synonyms (0.0-1.0)
- Considers word stems and synonyms
- Higher = better meaning preservation

**BERTScore-F1**: Contextual embedding similarity (0.0-1.0)
- Uses BERT to measure semantic similarity
- Higher = better semantic match
- **NEW**: Captures meaning even with different wording

### Example Output

```
==================================================
EVALUATION RESULTS
==================================================

BLEU Scores:
  BLEU-1: 0.3299
  BLEU-2: 0.3025
  BLEU-3: 0.2874
  BLEU-4: 0.1850

ROUGE Scores:
  ROUGE-1: 0.4942
  ROUGE-2: 0.4309
  ROUGE-L: 0.3350

METEOR Score:
  METEOR: 0.4500

BERTScore (Semantic Similarity):
  PRECISION: 0.8750
  RECALL: 0.8650
  F1: 0.8700

==================================================
```

### Disable BERTScore

If you don't want to use BERTScore (e.g., to save time), edit `config.yaml`:

```yaml
evaluation:
  use_bertscore: false
```

### Phase 1 vs Phase 2 Evaluation

**Phase 1 (Baseline - No Reflective Agent):**
```bash
python evaluate.py --checkpoint ./outputs/checkpoint-XXXX --no_reflective_agent --num_samples 20
```

**Phase 2 (With Reflective Agent - Not Evaluated Yet):**
```bash
# First enable in config.yaml:
# reflective_agent:
#   enabled: true

python evaluate.py --checkpoint ./outputs/checkpoint-XXXX --num_samples 20
```

### Performance Tips

1. **Quick testing**: Use `--num_samples 20` to test on 20 samples (~2-3 minutes)
2. **Fast mode**: Add `--fast_mode` for greedy decoding (30-50% faster)
3. **Disable BERTScore**: Set `use_bertscore: false` in config if not needed
4. **Full evaluation**: Remove `--num_samples` for complete test set (~45-60 minutes)

### Expected Scores (Phase 1 Baseline)

Based on compact structures only (no reflective agent):

- BLEU-4: ~0.185
- ROUGE-L: ~0.335
- METEOR: ~0.450
- BERTScore-F1: ~0.85-0.87 (expected)
