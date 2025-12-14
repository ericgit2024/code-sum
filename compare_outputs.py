"""
Compare model outputs with reference summaries to diagnose issues.
"""

import os
import sys
import yaml
import json
import argparse
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_loader import load_codesearchnet_dataset
from src.data.preprocessor import DataPreprocessor
from src.rag.rag_system import RAGSystem
from src.model.model_loader import load_model
from src.agent.reflective_agent import ReflectiveAgent
from src.model.inference import InferencePipeline


def compare_samples(config_path='config.yaml', checkpoint_path=None, num_samples=5, 
                   use_reflective_agent=True, hf_token=None):
    """
    Generate summaries and compare with references.
    
    Args:
        config_path: Path to config file
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to compare
        use_reflective_agent: Whether to use reflective agent
        hf_token: HuggingFace token
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print("Loading dataset...")
    _, _, test_data = load_codesearchnet_dataset(config)
    
    # Limit samples
    test_data = test_data[:num_samples]
    
    # Load RAG system
    print("Loading RAG system...")
    rag_system = RAGSystem(config)
    try:
        rag_system.load_index()
    except:
        print("Warning: Could not load RAG index")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        config,
        hf_token or os.getenv('HF_TOKEN'),
        for_training=False,
        checkpoint_path=checkpoint_path
    )
    
    # Initialize reflective agent
    reflective_agent = ReflectiveAgent(model, tokenizer, config, eval_mode=True)
    
    # Initialize inference pipeline
    inference_pipeline = InferencePipeline(
        model, tokenizer, rag_system, preprocessor,
        reflective_agent, config
    )
    
    # Generate and compare
    print(f"\n{'='*80}")
    print(f"COMPARING {num_samples} SAMPLES")
    print(f"Reflective Agent: {'ENABLED' if use_reflective_agent else 'DISABLED'}")
    print(f"{'='*80}\n")
    
    comparisons = []
    
    for i, sample in enumerate(test_data):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{num_samples}")
        print(f"{'='*80}\n")
        
        # Get code and reference
        code = sample['code']
        reference = sample.get('docstring', sample.get('summary', ''))
        
        # Generate summary
        result = inference_pipeline.predict_single(
            code,
            use_reflective_agent=use_reflective_agent
        )
        
        # Print comparison
        print("CODE:")
        print("-" * 80)
        print(code[:500] + ("..." if len(code) > 500 else ""))
        print()
        
        print("REFERENCE SUMMARY (Ground Truth):")
        print("-" * 80)
        print(reference)
        print()
        
        print("GENERATED SUMMARY (Initial):")
        print("-" * 80)
        print(result['initial_summary'])
        print()
        
        if use_reflective_agent and result['iterations'] > 0:
            print("GENERATED SUMMARY (Final - After Reflective Agent):")
            print("-" * 80)
            print(result['final_summary'])
            print(f"\nIterations: {result['iterations']}")
            print()
        
        print("PROMPT USED:")
        print("-" * 80)
        # Reconstruct prompt
        structures = preprocessor.extract_structures(code)
        rag_enabled = config.get('rag', {}).get('enabled', False)
        if rag_enabled:
            retrieved = rag_system.retrieve(code)
            rag_context = rag_system.format_rag_context(retrieved)
        else:
            rag_context = ""
        prompt = preprocessor.format_prompt(code, structures, rag_context)
        print(prompt[:800] + ("..." if len(prompt) > 800 else ""))
        print()
        
        # Store comparison
        comparisons.append({
            'sample_id': i + 1,
            'code': code,
            'reference': reference,
            'initial_summary': result['initial_summary'],
            'final_summary': result['final_summary'],
            'iterations': result['iterations'],
            'structures': structures
        })
        
        print(f"\n{'='*80}\n")
    
    # Save comparisons to file
    output_file = 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(comparisons, f, indent=2)
    
    print(f"\nComparisons saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples compared: {len(comparisons)}")
    print(f"Average reference length: {sum(len(c['reference']) for c in comparisons) / len(comparisons):.1f} chars")
    print(f"Average generated length: {sum(len(c['final_summary']) for c in comparisons) / len(comparisons):.1f} chars")
    if use_reflective_agent:
        avg_iterations = sum(c['iterations'] for c in comparisons) / len(comparisons)
        print(f"Average iterations: {avg_iterations:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare model outputs with references')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token')
    parser.add_argument('--no_reflective_agent', action='store_true',
                       help='Disable reflective agent')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to compare (default: 5)')
    
    args = parser.parse_args()
    
    # Load environment
    load_dotenv()
    
    compare_samples(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        use_reflective_agent=not args.no_reflective_agent,
        hf_token=args.hf_token
    )
