"""
Evaluation script for trained model.
"""

import os
import yaml
import argparse
import json
from dotenv import load_dotenv

from src.data.dataset_loader import load_codesearchnet_dataset
from src.data.preprocessor import DataPreprocessor
from src.rag.rag_system import RAGSystem
from src.model.model_loader import load_model
from src.agent.reflective_agent import ReflectiveAgent
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
    parser.add_argument('--no_reflective_agent', action='store_true',
                       help='Disable reflective agent')
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
    
    # Step 2: Load RAG index
    print("\n[2/6] Loading RAG index...")
    rag_system = RAGSystem(config)
    rag_system.load_index()
    
    # Step 3: Initialize preprocessor
    print("\n[3/6] Initializing preprocessor...")
    preprocessor = DataPreprocessor(config)
    
    # Step 4: Load model
    print("\n[4/6] Loading trained model...")
    model, tokenizer = load_model(
        config,
        hf_token,
        for_training=False,
        checkpoint_path=args.checkpoint
    )
    
    # Step 5: Initialize reflective agent
    print("\n[5/6] Initializing reflective agent...")
    reflective_agent = ReflectiveAgent(model, tokenizer, config)
    
    # Initialize inference pipeline
    inference_pipeline = InferencePipeline(
        model, tokenizer, rag_system, preprocessor,
        reflective_agent, config
    )
    
    # Step 6: Generate predictions
    print("\n[6/6] Generating predictions...")
    use_reflective = not args.no_reflective_agent
    predictions = inference_pipeline.predict_batch(test_data, use_reflective_agent=use_reflective)
    
    # Evaluate
    print("\nCalculating metrics...")
    evaluator = EvaluationMetrics(config)
    
    references = [p['reference'] for p in predictions]
    hypotheses = [p['final_summary'] for p in predictions]
    
    results = evaluator.evaluate_batch(references, hypotheses)
    
    # Add metadata
    results['num_samples'] = len(predictions)
    results['reflective_agent_enabled'] = use_reflective
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
