"""
Phase 1 RL Training Script.

Small-scale RL training (500 samples, 3-5 epochs) to validate the approach
before full-scale Phase 2 training.
"""

import os
import sys
import yaml
import json
import random
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_loader import load_codesearchnet_dataset
from src.data.preprocessor import DataPreprocessor
from src.model.model_loader import load_model
from src.rl.ppo_trainer import DocstringPPOTrainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Phase 1 RL Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default='./outputs/final_model',
                       help='Path to supervised model checkpoint (starting point)')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of training samples for Phase 1')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--output_dir', type=str, default='./outputs/rl_phase1',
                       help='Output directory for RL checkpoints')
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override RL config with command-line args
    if 'rl' not in config:
        config['rl'] = {}
    config['rl']['phase1_samples'] = args.num_samples
    config['rl']['num_epochs'] = args.num_epochs
    
    print("\n" + "="*70)
    print("PHASE 1: RL TRAINING WITH EXECUTION-BASED REWARDS")
    print("="*70)
    print(f"\nTraining Configuration:")
    print(f"  Samples: {args.num_samples}")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {config['rl'].get('batch_size', 4)}")
    print(f"  Learning rate: {config['rl'].get('learning_rate', 1e-5)}")
    print(f"  Starting checkpoint: {args.checkpoint}")
    print(f"  Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print("\n[1/5] Loading dataset...")
    train_data, val_data, test_data = load_codesearchnet_dataset(config)
    
    # Subsample for Phase 1
    random.seed(42)
    train_subset = random.sample(train_data, min(args.num_samples, len(train_data)))
    val_subset = random.sample(val_data, min(100, len(val_data)))
    
    print(f"  Training samples: {len(train_subset)}")
    print(f"  Validation samples: {len(val_subset)}")
    
    # Load model
    print("\n[2/5] Loading supervised model checkpoint...")
    model, tokenizer = load_model(
        config, hf_token, for_training=True, checkpoint_path=args.checkpoint
    )
    print("  Model loaded successfully!")
    
    # Initialize preprocessor
    print("\n[3/5] Initializing preprocessor...")
    preprocessor = DataPreprocessor(config)
    
    # Initialize PPO trainer
    print("\n[4/5] Initializing PPO trainer...")
    ppo_trainer = DocstringPPOTrainer(model, tokenizer, config, preprocessor)
    
    # Training loop
    print("\n[5/5] Starting RL training...")
    print("="*70)
    
    batch_size = config['rl'].get('batch_size', 4)
    log_interval = config['rl'].get('log_interval', 10)
    
    training_history = {
        'epochs': [],
        'batches': [],
        'rewards': [],
        'losses': [],
        'kl_divergences': []
    }
    
    for epoch in range(args.num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{args.num_epochs}")
        print(f"{'='*70}\n")
        
        # Shuffle training data
        random.shuffle(train_subset)
        
        epoch_metrics = {
            'rewards': [],
            'losses': [],
            'kl_divs': []
        }
        
        # Training batches
        num_batches = len(train_subset) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch = train_subset[start_idx:end_idx]
            
            # Extract code and references
            code_batch = [sample['code'] for sample in batch]
            reference_batch = [sample['docstring'] for sample in batch]
            
            # Training step
            metrics = ppo_trainer.train_step(code_batch, reference_batch)
            
            # Track metrics
            epoch_metrics['rewards'].append(metrics['mean_reward'])
            epoch_metrics['losses'].append(metrics['ppo_loss'])
            epoch_metrics['kl_divs'].append(metrics['kl_divergence'])
            
            training_history['batches'].append(epoch * num_batches + batch_idx)
            training_history['rewards'].append(metrics['mean_reward'])
            training_history['losses'].append(metrics['ppo_loss'])
            training_history['kl_divergences'].append(metrics['kl_divergence'])
            
            # Log progress
            if (batch_idx + 1) % log_interval == 0 or batch_idx == 0:
                print(f"Batch {batch_idx + 1}/{num_batches}:")
                print(f"  Reward: {metrics['mean_reward']:.3f} "
                      f"(±{metrics['std_reward']:.3f}) "
                      f"[{metrics['min_reward']:.3f}, {metrics['max_reward']:.3f}]")
                print(f"  PPO Loss: {metrics['ppo_loss']:.4f}")
                print(f"  KL Div: {metrics['kl_divergence']:.4f}")
        
        # Epoch summary
        avg_reward = sum(epoch_metrics['rewards']) / len(epoch_metrics['rewards'])
        avg_loss = sum(epoch_metrics['losses']) / len(epoch_metrics['losses'])
        avg_kl = sum(epoch_metrics['kl_divs']) / len(epoch_metrics['kl_divs'])
        
        training_history['epochs'].append({
            'epoch': epoch + 1,
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
            'avg_kl': avg_kl
        })
        
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1} SUMMARY")
        print(f"{'='*70}")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Average KL Divergence: {avg_kl:.4f}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}")
        ppo_trainer.save_checkpoint(checkpoint_dir)
        print(f"  Checkpoint saved: {checkpoint_dir}")
    
    # Save final model
    final_dir = os.path.join(args.output_dir, "final_model")
    ppo_trainer.save_checkpoint(final_dir)
    
    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\nTraining history saved: {history_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("PHASE 1 TRAINING COMPLETE!")
    print("="*70)
    
    initial_reward = training_history['epochs'][0]['avg_reward']
    final_reward = training_history['epochs'][-1]['avg_reward']
    improvement = final_reward - initial_reward
    
    print(f"\n📊 Training Summary:")
    print(f"  Initial Reward: {initial_reward:.3f}")
    print(f"  Final Reward: {final_reward:.3f}")
    print(f"  Improvement: {improvement:+.3f} ({improvement/initial_reward*100:+.1f}%)")
    
    print(f"\n📁 Outputs:")
    print(f"  Final model: {final_dir}")
    print(f"  Training history: {history_path}")
    print(f"  Checkpoints: {args.output_dir}/checkpoint_epoch_*")
    
    print(f"\n🎯 Next Steps:")
    print(f"  1. Analyze rewards: python analyze_rewards.py --checkpoint {final_dir}")
    print(f"  2. Evaluate metrics: python evaluate.py --checkpoint {final_dir}")
    print(f"  3. Compare with baseline: python compare_outputs.py")
    print(f"  4. If results are good, proceed to Phase 2 (full dataset training)")


if __name__ == "__main__":
    main()
