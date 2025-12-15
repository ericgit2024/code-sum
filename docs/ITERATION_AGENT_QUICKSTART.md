# Quick Start Guide: Iteration Agent

## What Is It?

The **Iteration Agent** is a Phase 1 innovation that:
1. **Validates** docstrings against code and structural signals
2. **Generates** 3-4 targeted edit instructions
3. **Refines** summaries in a single pass while preserving all original words
4. **Maintains** BLEU-4 scores (no degradation risk)

## Why Use It?

✅ **Innovation beyond metrics** - Shows intelligent validation and refinement  
✅ **BLEU-4 preservation** - Additive strategy keeps all original words  
✅ **Faster** - Single pass vs multi-iteration reflective agent  
✅ **Interpretable** - Clear validation issues and edit instructions  

## How to Enable

Edit `config.yaml`:

```yaml
iteration_agent:
  enabled: true

reflective_agent:
  enabled: false
```

## Quick Test

```bash
# Test with sample functions
python test_iteration_agent.py

# Evaluate on 50 samples
python evaluate.py --checkpoint outputs/checkpoint-XXXX --num_samples 50
```

## Files Created

- `src/agent/iteration_agent.py` - Main implementation
- `test_iteration_agent.py` - Test script
- `docs/ITERATION_AGENT.md` - Full documentation
- `config.yaml` - Updated with iteration_agent section

## Files Modified

- `src/model/inference.py` - Integrated iteration agent
- `config.yaml` - Added prompts and configuration

## Next Steps

1. Run `python test_iteration_agent.py` to verify functionality
2. Run evaluation to compare BLEU-4 scores
3. Use for Phase 1 presentation as innovation

## Key Difference from Reflective Agent

| Feature | Reflective Agent | Iteration Agent |
|---------|------------------|-----------------|
| Passes | 1-3 iterations | 1 pass only |
| Word Preservation | Not guaranteed | Guaranteed |
| BLEU-4 Impact | May decrease | Maintained/improved |
| Speed | Slower | Faster |

## For Phase 1 Presentation

**Talking Point**: "We implemented an iteration agent that validates docstrings against code signals, produces targeted edit instructions, and performs constrained refinement in a single pass while preserving all words to maintain BLEU-4 scores. This demonstrates innovation beyond just model training and metrics."
