"""
Analyze reward distributions and identify improvement areas.

This script:
1. Loads a trained model
2. Generates docstrings for test samples
3. Computes rewards for generated and reference docstrings
4. Generates visualizations and statistics
5. Identifies low-reward examples for analysis
"""

import os
import sys
import yaml
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_loader import load_codesearchnet_dataset
from src.data.preprocessor import DataPreprocessor
from src.model.model_loader import load_model
from src.model.inference import InferencePipeline
from src.rag.rag_system import RAGSystem
from src.agent.reflective_agent import ReflectiveAgent
from src.rl.reward_function import RewardFunction


def plot_reward_distribution(rewards: List[float], title: str, output_path: str):
    """Plot histogram of reward distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.3f}')
    plt.axvline(np.median(rewards), color='green', linestyle='--',
                label=f'Median: {np.median(rewards):.3f}')
    plt.xlabel('Reward Score')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_component_breakdown(component_means: Dict[str, float], output_path: str):
    """Plot bar chart of average component scores."""
    components = list(component_means.keys())
    scores = list(component_means.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(components, scores, alpha=0.7, edgecolor='black')
    
    # Color bars based on score
    for i, (bar, score) in enumerate(zip(bars, scores)):
        if score >= 0.8:
            bar.set_color('green')
        elif score >= 0.6:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Good (≥0.8)')
    plt.axhline(0.6, color='orange', linestyle='--', alpha=0.5, label='Fair (≥0.6)')
    plt.xlabel('Reward Component')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Reward Component')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {output_path}")


def save_examples(predictions: List[Dict], output_path: str, num_examples: int = 10):
    """Save low-reward and high-reward examples to file."""
    # Sort by reward
    sorted_preds = sorted(predictions, key=lambda x: x['reward'])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("REWARD ANALYSIS - EXAMPLE DOCSTRINGS\n")
        f.write("="*80 + "\n\n")
        
        # Low-reward examples
        f.write(f"\n{'='*80}\n")
        f.write(f"LOW-REWARD EXAMPLES (Bottom {num_examples})\n")
        f.write(f"{'='*80}\n\n")
        
        for i, pred in enumerate(sorted_preds[:num_examples], 1):
            f.write(f"\n--- Example {i} (Reward: {pred['reward']:.3f}) ---\n\n")
            f.write(f"Code:\n{pred['code']}\n\n")
            f.write(f"Generated Docstring:\n{pred['generated']}\n\n")
            f.write(f"Reference Docstring:\n{pred['reference']}\n\n")
            f.write(f"Component Breakdown:\n")
            for component, score in pred['breakdown'].items():
                f.write(f"  - {component}: {score:.3f}\n")
            f.write("\n" + "-"*80 + "\n")
        
        # High-reward examples
        f.write(f"\n{'='*80}\n")
        f.write(f"HIGH-REWARD EXAMPLES (Top {num_examples})\n")
        f.write(f"{'='*80}\n\n")
        
        for i, pred in enumerate(sorted_preds[-num_examples:], 1):
            f.write(f"\n--- Example {i} (Reward: {pred['reward']:.3f}) ---\n\n")
            f.write(f"Code:\n{pred['code']}\n\n")
            f.write(f"Generated Docstring:\n{pred['generated']}\n\n")
            f.write(f"Reference Docstring:\n{pred['reference']}\n\n")
            f.write(f"Component Breakdown:\n")
            for component, score in pred['breakdown'].items():
                f.write(f"  - {component}: {score:.3f}\n")
            f.write("\n" + "-"*80 + "\n")
    
    print(f"Saved examples: {output_path}")


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(description='Analyze reward distributions')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of test samples to analyze')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load environment
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*70)
    print("REWARD ANALYSIS PIPELINE")
    print("="*70)
    
    # Load dataset
    print("\n[1/7] Loading test dataset...")
    _, _, test_data = load_codesearchnet_dataset(config)
    test_data = test_data[:args.num_samples]
    print(f"Loaded {len(test_data)} test samples")
    
    # Load model
    print("\n[2/7] Loading model...")
    model, tokenizer = load_model(
        config, hf_token, for_training=False, checkpoint_path=args.checkpoint
    )
    
    # Initialize components
    print("\n[3/7] Initializing components...")
    rag_system = RAGSystem(config)
    preprocessor = DataPreprocessor(config)
    reflective_agent = ReflectiveAgent(model, tokenizer, config, eval_mode=False)
    
    inference_pipeline = InferencePipeline(
        model, tokenizer, rag_system, preprocessor, reflective_agent, config
    )
    
    # Initialize reward function
    reward_fn = RewardFunction(config)
    
    # Generate docstrings
    print("\n[4/7] Generating docstrings...")
    predictions = []
    
    for i, sample in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(test_data)} samples...")
        
        # Generate docstring (without reflective agent for speed)
        result = inference_pipeline.predict_single(
            sample['code'], 
            use_reflective_agent=False
        )
        
        predictions.append({
            'code': sample['code'],
            'generated': result['final_summary'],
            'reference': sample['docstring']
        })
    
    # Compute rewards
    print("\n[5/7] Computing rewards...")
    
    # For generated docstrings
    generated_rewards = []
    generated_breakdowns = []
    
    for pred in predictions:
        reward, breakdown = reward_fn.compute_reward(
            pred['code'], pred['generated'], pred['reference']
        )
        generated_rewards.append(reward)
        generated_breakdowns.append(breakdown)
        pred['reward'] = reward
        pred['breakdown'] = breakdown
    
    # For reference docstrings (baseline)
    reference_rewards = []
    reference_breakdowns = []
    
    for pred in predictions:
        reward, breakdown = reward_fn.compute_reward(
            pred['code'], pred['reference'], pred['reference']
        )
        reference_rewards.append(reward)
        reference_breakdowns.append(breakdown)
    
    # Compute statistics
    print("\n[6/7] Computing statistics...")
    
    # Component-wise averages
    component_names = list(generated_breakdowns[0].keys())
    generated_component_means = {
        comp: np.mean([b[comp] for b in generated_breakdowns])
        for comp in component_names
    }
    reference_component_means = {
        comp: np.mean([b[comp] for b in reference_breakdowns])
        for comp in component_names
    }
    
    # Overall statistics
    stats = {
        'num_samples': len(predictions),
        'generated': {
            'mean': float(np.mean(generated_rewards)),
            'median': float(np.median(generated_rewards)),
            'std': float(np.std(generated_rewards)),
            'min': float(np.min(generated_rewards)),
            'max': float(np.max(generated_rewards)),
            'percentiles': {
                '25': float(np.percentile(generated_rewards, 25)),
                '50': float(np.percentile(generated_rewards, 50)),
                '75': float(np.percentile(generated_rewards, 75)),
                '90': float(np.percentile(generated_rewards, 90))
            },
            'component_means': generated_component_means
        },
        'reference': {
            'mean': float(np.mean(reference_rewards)),
            'median': float(np.median(reference_rewards)),
            'std': float(np.std(reference_rewards)),
            'min': float(np.min(reference_rewards)),
            'max': float(np.max(reference_rewards)),
            'component_means': reference_component_means
        },
        'gap': {
            'mean': float(np.mean(reference_rewards) - np.mean(generated_rewards)),
            'median': float(np.median(reference_rewards) - np.median(generated_rewards))
        }
    }
    
    # Print statistics
    print("\n" + "="*70)
    print("REWARD STATISTICS")
    print("="*70)
    
    print(f"\n📊 Generated Docstrings:")
    print(f"  Mean:   {stats['generated']['mean']:.3f}")
    print(f"  Median: {stats['generated']['median']:.3f}")
    print(f"  Std:    {stats['generated']['std']:.3f}")
    print(f"  Range:  [{stats['generated']['min']:.3f}, {stats['generated']['max']:.3f}]")
    
    print(f"\n📚 Reference Docstrings:")
    print(f"  Mean:   {stats['reference']['mean']:.3f}")
    print(f"  Median: {stats['reference']['median']:.3f}")
    print(f"  Std:    {stats['reference']['std']:.3f}")
    print(f"  Range:  [{stats['reference']['min']:.3f}, {stats['reference']['max']:.3f}]")
    
    print(f"\n📉 Gap (Reference - Generated):")
    print(f"  Mean Gap:   {stats['gap']['mean']:.3f}")
    print(f"  Median Gap: {stats['gap']['median']:.3f}")
    
    print(f"\n🔍 Component-wise Comparison:")
    print(f"{'Component':<25} {'Generated':<12} {'Reference':<12} {'Gap':<10}")
    print("-" * 60)
    for comp in component_names:
        gen_score = generated_component_means[comp]
        ref_score = reference_component_means[comp]
        gap = ref_score - gen_score
        print(f"{comp:<25} {gen_score:<12.3f} {ref_score:<12.3f} {gap:<10.3f}")
    
    # Identify weakest components
    component_gaps = {
        comp: reference_component_means[comp] - generated_component_means[comp]
        for comp in component_names
    }
    sorted_gaps = sorted(component_gaps.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n⚠️  Weakest Components (largest gaps):")
    for i, (comp, gap) in enumerate(sorted_gaps[:3], 1):
        print(f"  {i}. {comp}: {gap:.3f} gap")
    
    # Generate visualizations
    print("\n[7/7] Generating visualizations...")
    
    plot_reward_distribution(
        generated_rewards,
        "Generated Docstring Reward Distribution",
        os.path.join(args.output_dir, "reward_distribution_generated.png")
    )
    
    plot_reward_distribution(
        reference_rewards,
        "Reference Docstring Reward Distribution",
        os.path.join(args.output_dir, "reward_distribution_reference.png")
    )
    
    plot_component_breakdown(
        generated_component_means,
        os.path.join(args.output_dir, "component_breakdown_generated.png")
    )
    
    plot_component_breakdown(
        reference_component_means,
        os.path.join(args.output_dir, "component_breakdown_reference.png")
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, "reward_analysis.json")
    with open(results_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics: {results_path}")
    
    # Save examples
    examples_path = os.path.join(args.output_dir, "reward_examples.txt")
    save_examples(predictions, examples_path, num_examples=10)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {args.output_dir}/")
    print("\n📌 Key Findings:")
    print(f"  - Current model reward: {stats['generated']['mean']:.3f}")
    print(f"  - Target reward (reference): {stats['reference']['mean']:.3f}")
    print(f"  - Improvement potential: {stats['gap']['mean']:.3f}")
    print(f"\n🎯 Focus RL training on: {', '.join([comp for comp, _ in sorted_gaps[:3]])}")


if __name__ == "__main__":
    main()
