"""
Interactive inference script for generating summaries.
"""

import os
import sys
import yaml
import argparse
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.preprocessor import DataPreprocessor
from src.rag.rag_system import RAGSystem
from src.model.model_loader import load_model
from src.agent.entity_refinement_agent import EntityRefinementAgent
from src.model.inference import InferencePipeline


def main():
    """Main inference function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate code summary')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--code', type=str, default=None,
                       help='Python code to summarize')
    parser.add_argument('--code_file', type=str, default=None,
                       help='Path to Python file to summarize')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token')
    parser.add_argument('--no_entity_agent', action='store_true',
                       help='Disable entity refinement agent')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    
    # Get code
    if args.code:
        code = args.code
    elif args.code_file:
        with open(args.code_file, 'r') as f:
            code = f.read()
    else:
        print("Error: Provide either --code or --code_file")
        return
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("CODE SUMMARIZATION - INFERENCE")
    print("="*60)
    
    # Load RAG index
    print("\nLoading RAG index...")
    rag_system = RAGSystem(config)
    rag_system.load_index()
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = DataPreprocessor(config)
    
    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        config,
        hf_token,
        for_training=False,
        checkpoint_path=args.checkpoint
    )
    
    # Initialize entity refinement agent
    print("Initializing entity refinement agent...")
    entity_agent = EntityRefinementAgent(model, tokenizer, config)
    
    # Initialize inference pipeline
    inference_pipeline = InferencePipeline(
        model, tokenizer, rag_system, preprocessor,
        entity_agent, config
    )
    
    # Generate summary
    print("\nGenerating summary...")
    print("-"*60)
    
    use_entity_agent = not args.no_entity_agent
    result = inference_pipeline.predict_single(code, use_entity_agent=use_entity_agent)
    
    # Display results
    print("\nCODE:")
    print(code)
    print("\n" + "-"*60)
    
    print("\nINITIAL SUMMARY:")
    print(result['initial_summary'])
    
    if use_entity_agent:
        print("\n" + "-"*60)
        print(f"\nFINAL SUMMARY (after {result['iterations']} iteration(s)):")
        print(result['final_summary'])
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
