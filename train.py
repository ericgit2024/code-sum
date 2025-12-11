"""
Main training script for code summarization project.
"""

import os
import sys
import yaml
import argparse
from dotenv import load_dotenv

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_loader import load_codesearchnet_dataset
from src.data.preprocessor import DataPreprocessor
from src.rag.rag_system import RAGSystem
from src.model.model_loader import load_model
from src.model.trainer import train_model


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train code summarization model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='HuggingFace token (or set HF_TOKEN env var)')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get HF token
    hf_token = args.hf_token or os.getenv('HF_TOKEN')
    if not hf_token:
        print("WARNING: No HuggingFace token provided. Set HF_TOKEN environment variable or use --hf_token argument.")
        print("You can continue, but you may not be able to access gated models like Gemma.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("CODE SUMMARIZATION - TRAINING PIPELINE")
    print("="*60)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading CodeSearchNet dataset...")
    train_data, val_data, test_data = load_codesearchnet_dataset(config)
    
    # Step 2: Build RAG index
    print("\n[2/6] Building RAG index...")
    rag_system = RAGSystem(config)
    rag_system.build_index(train_data)
    rag_system.save_index()
    
    # Step 3: Preprocess data
    print("\n[3/6] Preprocessing training data...")
    preprocessor = DataPreprocessor(config)
    
    # Preprocess training data with RAG
    train_preprocessed = preprocessor.preprocess_dataset(train_data, rag_system)
    
    # Preprocess validation data with RAG
    print("\n[4/6] Preprocessing validation data...")
    val_preprocessed = preprocessor.preprocess_dataset(val_data, rag_system)
    
    # Step 4: Load model
    print("\n[5/6] Loading Gemma 2B model with LoRA...")
    model, tokenizer = load_model(config, hf_token, for_training=True)
    
    # Step 5: Train model
    print("\n[6/6] Training model...")
    train_model(model, tokenizer, config, train_preprocessed, val_preprocessed)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {config['training']['output_dir']}/final_model")
    print(f"RAG index saved to: {config['rag']['index_path']}")
    print("\nNext steps:")
    print("1. Run evaluation: python evaluate.py")
    print("2. Run inference: python run_inference.py --code 'your_code_here'")


if __name__ == "__main__":
    main()
