# RL-Based Docstring Generation: Solution Overview

## Problem Statement

While supervised fine-tuning of large language models has shown promising results in code summarization tasks, current approaches suffer from a critical limitation: they optimize for syntactic similarity to reference docstrings rather than behavioral accuracy. Our analysis of the Gemma 2B model fine-tuned on the CodeSearchNet dataset reveals that the model generates grammatically correct and stylistically appropriate docstrings, yet frequently omits crucial behavioral information such as parameter descriptions, return value specifications, and control flow logic (if/else conditions, loops, error handling). This occurs because traditional supervised learning with BLEU, ROUGE, and METEOR metrics rewards surface-level text similarity without verifying whether the generated docstring accurately describes what the code actually does. For instance, a function with parameters `price` and `discount_percent` that caps discounts at 50% might receive a docstring like "Calculates discount" which, while not incorrect, fails to mention the parameters or the capping behavior. This behavioral incompleteness limits the practical utility of generated docstrings for developers who rely on them to understand function behavior without reading the implementation. The fundamental issue is that supervised learning lacks a mechanism to evaluate whether the docstring captures the semantic behavior encoded in the code structure, leading to outputs that are syntactically perfect but semantically incomplete.

## Proposed Solution

We propose a novel reinforcement learning framework that addresses behavioral incompleteness through execution-based reward signals derived from automated code analysis. Our approach employs Proximal Policy Optimization (PPO), the industry-standard RL algorithm used by state-of-the-art systems like ChatGPT and Claude, to fine-tune the supervised model using a custom multi-component reward function. This reward function leverages Abstract Syntax Tree (AST) parsing to extract structural information from the code—including parameters, return statements, control flow constructs (if/else, loops, try/except), and function calls—and evaluates whether the generated docstring accurately describes these behavioral elements. The reward is computed as a weighted combination of six components: (1) parameter coverage (25% weight) verifies all function parameters are mentioned, (2) return value mention (25%) ensures return behavior is described, (3) control flow coverage (20%) checks if conditional logic and loops are explained, (4) naturalness (15%) penalizes code syntax in favor of natural language, (5) fluency (10%) maintains readability through BLEU-based scoring, and (6) hallucination penalty (5%) prevents mentioning non-existent features. By optimizing for these execution-based rewards rather than reference similarity, the model learns to generate docstrings that are not only fluent and natural but also behaviorally complete and accurate. The PPO algorithm with KL divergence penalty ensures stable training by preventing the model from drifting too far from its supervised baseline, preserving the quality gains from initial fine-tuning while incrementally improving behavioral accuracy. This approach eliminates the need for additional human annotation, as rewards are computed automatically from code structure, making it scalable and cost-effective.

## Workflow

The complete workflow consists of two phases: diagnostic analysis followed by reinforcement learning training. In Phase 1 (Diagnostic Analysis), we first validate the reward function using a comprehensive test suite with seven diverse code samples, ensuring that high-quality docstrings (those mentioning parameters, returns, and control flow) receive scores of 0.7-0.9 while incomplete docstrings score 0.2-0.5, demonstrating clear discrimination. We then analyze the current supervised model by generating docstrings for 100 test samples and computing their rewards, producing visualizations that identify the weakest components—typically control flow coverage and parameter mentions, which often show gaps of 0.2-0.3 points compared to reference docstrings. This analysis reveals that while the model achieves an average reward of approximately 0.55, reference docstrings score around 0.81, indicating substantial room for improvement. Phase 2 (RL Training) begins with small-scale validation on 500 samples over 5 epochs (~2-3 hours on T4 GPU) to verify the approach before full-scale training. The PPO trainer loads the supervised model as initialization, generates docstrings using the current policy, computes execution-based rewards for each generation, and applies policy gradient updates with KL penalty to maximize rewards while maintaining stability. Training progress is monitored through three key metrics: reward scores (expected to increase from 0.55 to 0.65-0.70), KL divergence (must stay below 0.02 to prevent model drift), and PPO loss (should decrease over epochs). After validation, Phase 2 proceeds to full-scale training on 5000 samples over 10 epochs, targeting final rewards of 0.75-0.85 and BLEU scores of 0.23-0.27, which would represent state-of-the-art performance. Post-training evaluation compares the RL-trained model against the supervised baseline using both traditional metrics (BLEU, ROUGE, METEOR) and the execution-based reward components, with particular attention to improvements in the previously identified weak areas. The workflow concludes with qualitative inspection of generated docstrings to verify that the model now consistently includes parameter descriptions, return value specifications, and control flow explanations that were previously missing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RL Training Architecture                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Training Data   │
│  (CodeSearchNet) │
│   500 samples    │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           PPO Training Loop                              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        Iteration t                                 │  │
│  │                                                                    │  │
│  │  ┌──────────────┐         ┌──────────────────────────────────┐   │  │
│  │  │    Code      │────────▶│   Data Preprocessor              │   │  │
│  │  │   Sample     │         │  • AST Extraction                │   │  │
│  │  └──────────────┘         │  • Compact Structure Summary     │   │  │
│  │                           │  • Prompt Formatting             │   │  │
│  │                           └──────────┬───────────────────────┘   │  │
│  │                                      │                            │  │
│  │                                      ▼                            │  │
│  │                           ┌──────────────────────────────────┐   │  │
│  │                           │   Gemma 2B + LoRA (Policy π_θ)   │   │  │
│  │                           │  • 4-bit Quantization            │   │  │
│  │                           │  • Current Policy Parameters     │   │  │
│  │                           └──────────┬───────────────────────┘   │  │
│  │                                      │                            │  │
│  │                                      ▼                            │  │
│  │                           ┌──────────────────────────────────┐   │  │
│  │                           │   Generated Docstring            │   │  │
│  │                           │   "Calculates final price..."    │   │  │
│  │                           └──────────┬───────────────────────┘   │  │
│  │                                      │                            │  │
│  │                                      ▼                            │  │
│  │  ┌────────────────────────────────────────────────────────────┐  │  │
│  │  │              Execution-Based Reward Function               │  │  │
│  │  │                                                            │  │  │
│  │  │  ┌──────────────────┐  ┌──────────────────┐              │  │  │
│  │  │  │  AST Analysis    │  │  Docstring       │              │  │  │
│  │  │  │  • Parameters    │  │  Analysis        │              │  │  │
│  │  │  │  • Returns       │  │  • Text Parsing  │              │  │  │
│  │  │  │  • Control Flow  │  │  • Pattern Match │              │  │  │
│  │  │  └────────┬─────────┘  └────────┬─────────┘              │  │  │
│  │  │           │                     │                         │  │  │
│  │  │           └──────────┬──────────┘                         │  │  │
│  │  │                      ▼                                    │  │  │
│  │  │           ┌─────────────────────────┐                    │  │  │
│  │  │           │  Component Scores:      │                    │  │  │
│  │  │           │  • Parameter: 0.75      │                    │  │  │
│  │  │           │  • Return: 1.00         │                    │  │  │
│  │  │           │  • Control Flow: 0.60   │                    │  │  │
│  │  │           │  • Naturalness: 1.00    │                    │  │  │
│  │  │           │  • Fluency: 0.65        │                    │  │  │
│  │  │           │  • Hallucination: 1.00  │                    │  │  │
│  │  │           └────────────┬────────────┘                    │  │  │
│  │  │                        ▼                                 │  │  │
│  │  │           ┌─────────────────────────┐                    │  │  │
│  │  │           │  Weighted Reward: 0.82  │                    │  │  │
│  │  │           │  (Σ weight_i × score_i) │                    │  │  │
│  │  │           └────────────┬────────────┘                    │  │  │
│  │  └─────────────────────────┼─────────────────────────────────┘  │  │
│  │                            │                                     │  │
│  │                            ▼                                     │  │
│  │                 ┌──────────────────────────┐                     │  │
│  │                 │   PPO Update             │                     │  │
│  │                 │  • Advantage = r - b     │                     │  │
│  │                 │  • KL Penalty            │                     │  │
│  │                 │  • Policy Gradient       │                     │  │
│  │                 │  • Value Function Update │                     │  │
│  │                 └──────────┬───────────────┘                     │  │
│  │                            │                                     │  │
│  │                            ▼                                     │  │
│  │                 ┌──────────────────────────┐                     │  │
│  │                 │   Updated Policy π_θ'    │                     │  │
│  │                 │   (θ' = θ + Δθ)          │                     │  │
│  │                 └──────────────────────────┘                     │  │
│  │                                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  Repeat for N epochs until convergence or max iterations                │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Evaluation & Analysis                            │
│                                                                          │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐     │
│  │  Reward Analysis │  │  Metric Eval     │  │  Qualitative     │     │
│  │  • Distribution  │  │  • BLEU          │  │  • Examples      │     │
│  │  • Components    │  │  • ROUGE         │  │  • Comparison    │     │
│  │  • Improvement   │  │  • METEOR        │  │  • Error Analysis│     │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Final RL-Trained Model                           │
│                    (Behaviorally Accurate Docstrings)                    │
└─────────────────────────────────────────────────────────────────────────┘


Key Components:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Data Preprocessor: Extracts compact structural features (parameters, 
   control flow, return types) using AST analysis

2. Policy Network (Gemma 2B + LoRA): Current model that generates docstrings,
   updated via PPO to maximize rewards

3. Reward Function: Computes 6-component score based on code-docstring 
   alignment, no human labels required

4. PPO Trainer: Stable RL algorithm with KL penalty to prevent model drift
   from supervised baseline

5. Evaluation Suite: Multi-faceted analysis including reward distributions,
   traditional metrics, and qualitative examples
```

## Key Innovation

The core innovation lies in the **execution-based reward function** that automatically evaluates behavioral accuracy without human annotation. By parsing the code's Abstract Syntax Tree to extract ground-truth behavioral information (parameters, returns, control flow), we create a self-supervised signal that guides the model toward generating not just fluent text, but accurate descriptions of code behavior. This approach is novel in the docstring generation domain and represents a significant departure from traditional supervised learning that optimizes only for surface-level similarity to reference texts.
