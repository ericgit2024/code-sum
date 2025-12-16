"""
Evaluation script for trained model.
"""

import os
import sys
import yaml
import argparse
import json
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_loader import load_codesearchnet_dataset
from src.data.preprocessor import DataPreprocessor
from src.rag.rag_system import RAGSystem
from src.model.model_loader import load_model
from src.agent.critic_refinement_agent import CriticRefinementAgent
from src.model.inference import InferencePipeline
from src.evaluation.metrics import EvaluationMetrics


def main():
    """Main evaluation function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate code summarization model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token')
    parser.add_argument('--no_critic_agent', action='store_true',
                       help='Disable critic refinement agent')
    parser.add_argument('--fast_mode', action='store_true',
                       help='Enable fast mode (greedy decoding, reduced tokens)')
    parser.add_argument('--max_iterations', type=int, default=None,
                       help='Override max iterations for critic agent')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Limit number of test samples (e.g., 20 for quick testing)')
    parser.add_argument('--output', type=str, default='evaluation_results/results.json',
                       help='Output path for results')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("CODE SUMMARIZATION - EVALUATION PIPELINE")
    print("="*60)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading test dataset...")
    _, _, test_data = load_codesearchnet_dataset(config)
    
    # Limit samples if requested
    if args.num_samples:
        print(f"Limiting to {args.num_samples} samples for quick testing")
        test_data = test_data[:min(args.num_samples, len(test_data))]

    
    # Step 2: Load RAG index
    print("\n[2/6] Loading RAG index...")
    rag_system = RAGSystem(config)
    rag_system.load_index()
    
    # Step 3: Initialize preprocessor
    print("\n[3/6] Initializing preprocessor...")
    preprocessor = DataPreprocessor(config)
    
    # Apply fast mode settings if requested
    if args.fast_mode:
        print("Fast mode enabled: using greedy decoding and reduced token limits")
        config['summary_critic']['fast_mode'] = True
        config['summary_critic']['greedy_decoding'] = True
    
    # Override max iterations if specified
    if args.max_iterations is not None:
        print(f"Overriding max iterations: {args.max_iterations}")
        config['summary_critic']['max_iterations'] = args.max_iterations
    
    # Step 4: Load model
    print("\n[4/6] Loading trained model...")
    model, tokenizer = load_model(
        config,
        hf_token,
        for_training=False,
        checkpoint_path=args.checkpoint
    )
    
    # Step 5: Initialize critic refinement agent with eval mode
    print("\n[5/6] Initializing critic refinement agent...")
    critic_agent = CriticRefinementAgent(model, tokenizer, config, eval_mode=True)
    
    # Initialize inference pipeline
    inference_pipeline = InferencePipeline(
        model, tokenizer, rag_system, preprocessor,
        critic_agent, config
    )
    
    # Step 6: Generate predictions
    print("\n[6/6] Generating predictions...")
    
    # Check if critic agent is enabled
    use_critic = (not args.no_critic_agent and 
                     config['summary_critic'].get('enabled', True))
    
    if not use_critic:
        print("Critic agent disabled - using base model only")
    
    predictions = inference_pipeline.predict_batch(test_data, use_critic_agent=use_critic)
    
    # Evaluate
    print("\nCalculating metrics...")
    evaluator = EvaluationMetrics(config)
    
    references = [p['reference'] for p in predictions]
    hypotheses = [p['final_summary'] for p in predictions]
    
    results = evaluator.evaluate_batch(references, hypotheses)
    
    # Add metadata
    results['num_samples'] = len(predictions)
    results['critic_agent_enabled'] = use_critic
    results['checkpoint'] = args.checkpoint
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Save predictions
    predictions_path = args.output.replace('.json', '_predictions.json')
    with open(predictions_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Predictions saved to {predictions_path}")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
