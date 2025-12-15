# Phase 1: RL Training with Execution-Based Rewards - Usage Guide

## Overview

This guide walks you through Phase 1 of RL-based docstring training, which validates the approach on a small subset (500 samples) before full-scale training.

## Prerequisites

1. **Completed Supervised Training**: You should have a trained model from supervised learning at `./outputs/final_model`
2. **Dependencies Installed**: Run `pip install -r requirements.txt` to install TRL and matplotlib
3. **HuggingFace Token**: Set in `.env` file

## Phase 1 Workflow

### Step 1: Test the Reward Function

First, verify that the reward function works correctly:

```bash
python test_reward_function.py
```

**Expected Output**:
- 7 test cases with good vs bad docstrings
- Reward scores showing clear discrimination (difference > 0.2)
- Component breakdowns for each case

**What to Check**:
- Good docstrings should score 0.7-0.9
- Bad docstrings should score 0.2-0.5
- Component scores should make intuitive sense

---

### Step 2: Analyze Current Model Rewards

Before training, analyze where your current model needs improvement:

```bash
python analyze_rewards.py --checkpoint ./outputs/final_model --num_samples 100
```

**Expected Output**:
- Reward distribution plots in `evaluation_results/`
- Component breakdown showing weakest areas
- Example low/high reward docstrings
- Statistics comparing model vs reference rewards

**Key Metrics to Note**:
- **Current Model Mean Reward**: Likely 0.50-0.60
- **Reference Mean Reward**: Likely 0.75-0.85
- **Gap**: This is your improvement potential
- **Weakest Components**: Focus areas for RL training

**Example Output**:
```
📊 Generated Docstrings:
  Mean:   0.547
  Median: 0.553

📚 Reference Docstrings:
  Mean:   0.812
  Median: 0.825

📉 Gap: 0.265

⚠️  Weakest Components:
  1. control_flow: 0.312 gap
  2. parameter_coverage: 0.245 gap
  3. return_mention: 0.198 gap
```

---

### Step 3: Run Phase 1 RL Training

Train on 500 samples for 5 epochs (~2-3 hours on T4 GPU):

```bash
python train_rl_phase1.py \
  --checkpoint ./outputs/final_model \
  --num_samples 500 \
  --num_epochs 5 \
  --output_dir ./outputs/rl_phase1
```

**Training Progress**:
```
EPOCH 1/5
Batch 10/125:
  Reward: 0.563 (±0.142) [0.321, 0.789]
  PPO Loss: 0.0234
  KL Div: 0.0087

...

EPOCH 1 SUMMARY
  Average Reward: 0.571
  Average Loss: 0.0198
  Average KL Divergence: 0.0092
  Checkpoint saved: ./outputs/rl_phase1/checkpoint_epoch_1
```

**What to Monitor**:
- **Reward**: Should gradually increase (0.55 → 0.65+)
- **KL Divergence**: Should stay < 0.02 (prevents model drift)
- **Loss**: Should decrease over time

**Warning Signs**:
- KL divergence > 0.05: Model drifting too far, reduce learning rate
- Reward decreasing: Check reward function, may need to adjust weights
- Loss exploding: Reduce learning rate or gradient clipping

---

### Step 4: Analyze Post-Training Rewards

After training, analyze improvements:

```bash
python analyze_rewards.py --checkpoint ./outputs/rl_phase1/final_model --num_samples 100
```

**Expected Improvements**:
- Mean reward: 0.55 → 0.65-0.70 (+0.10-0.15)
- Weakest components should show largest gains
- Gap to reference should narrow

**Compare Before/After**:
```
Before RL:
  Mean Reward: 0.547
  control_flow: 0.423
  parameter_coverage: 0.512

After RL (Phase 1):
  Mean Reward: 0.673 (+0.126)
  control_flow: 0.598 (+0.175)
  parameter_coverage: 0.687 (+0.175)
```

---

### Step 5: Evaluate Standard Metrics

Check if BLEU/ROUGE improved:

```bash
python evaluate.py --checkpoint ./outputs/rl_phase1/final_model --no_reflective_agent
```

**Expected Results**:
- BLEU-4: 0.185 → 0.20-0.22 (+0.02-0.04)
- ROUGE-L: 0.335 → 0.35-0.37 (+0.02)
- METEOR: 0.450 → 0.47-0.49 (+0.02)

---

### Step 6: Qualitative Inspection

Generate docstrings for sample functions to verify quality:

```bash
python run_inference.py --checkpoint ./outputs/rl_phase1/final_model --code "def calculate_discount(price, discount_percent):
    if discount_percent > 50:
        discount_percent = 50
    return price * (1 - discount_percent / 100)"
```

**Before RL**:
```
"Calculates discount."
```

**After RL (Expected)**:
```
"Calculates the final price after applying a discount percentage. 
Caps the discount at 50% maximum to prevent excessive discounts. 
Returns the discounted price."
```

---

## Decision Point: Proceed to Phase 2?

### ✅ Proceed if:
- Reward improved by +0.10 or more
- BLEU improved by +0.02 or more
- Qualitative examples show better parameter/control flow coverage
- KL divergence stayed < 0.02 (model stable)

### ⚠️ Adjust if:
- Reward improved by < 0.05: Increase epochs or adjust reward weights
- KL divergence > 0.02: Reduce learning rate
- Specific components not improving: Increase their weights in config

### ❌ Reconsider if:
- Reward decreased or no improvement
- BLEU decreased (model quality degraded)
- KL divergence > 0.05 (model unstable)

---

## Phase 2: Full-Scale Training

If Phase 1 is successful, proceed to Phase 2:

```bash
python train_rl_phase2.py \
  --checkpoint ./outputs/rl_phase1/final_model \
  --num_samples 5000 \
  --num_epochs 10 \
  --output_dir ./outputs/rl_phase2
```

**Phase 2 Differences**:
- 5000 samples (10x more data)
- 10 epochs (more refinement)
- Expected runtime: ~20-30 hours on T4 GPU

**Expected Final Results**:
- Mean Reward: 0.75-0.85
- BLEU-4: 0.23-0.27 (SOTA range)
- ROUGE-L: 0.38-0.42
- METEOR: 0.50-0.54

---

## Troubleshooting

### Issue: Reward function gives unexpected scores

**Solution**: Run `test_reward_function.py` and manually inspect examples. Adjust weights in `config.yaml` if needed.

### Issue: Training is too slow

**Solution**: 
- Reduce `batch_size` to 2 (but increase `gradient_accumulation_steps` to 8)
- Reduce `num_samples` to 250 for faster iteration
- Use `--num_epochs 3` instead of 5

### Issue: KL divergence too high

**Solution**:
- Reduce `learning_rate` to 5e-6 in `config.yaml`
- Increase `kl_penalty` to 0.2
- Reduce `ppo_epochs` to 2

### Issue: Rewards not improving

**Solution**:
- Check if reward function aligns with your goals (run `test_reward_function.py`)
- Increase training epochs
- Adjust reward weights to emphasize weak components
- Verify model is loading correctly from checkpoint

---

## Files Generated

After Phase 1, you'll have:

```
outputs/rl_phase1/
├── checkpoint_epoch_1/      # Checkpoint after epoch 1
├── checkpoint_epoch_2/      # Checkpoint after epoch 2
├── ...
├── final_model/             # Final trained model
└── training_history.json    # Reward/loss curves

evaluation_results/
├── reward_analysis.json                    # Statistics
├── reward_distribution_generated.png       # Histogram
├── reward_distribution_reference.png       # Baseline
├── component_breakdown_generated.png       # Bar chart
├── component_breakdown_reference.png       # Baseline
└── reward_examples.txt                     # Low/high examples
```

---

## Next Steps

1. **Analyze Results**: Review all plots and statistics
2. **Compare Checkpoints**: Test different epoch checkpoints to find best
3. **Tune Hyperparameters**: Adjust based on Phase 1 findings
4. **Proceed to Phase 2**: Full-scale training with optimized settings
5. **Publish Results**: Document improvements for thesis/paper
