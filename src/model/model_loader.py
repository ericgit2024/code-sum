"""
Model loader for Gemma 2B with 4-bit quantization and LoRA.
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)
from typing import Dict, Tuple


class GemmaModelLoader:
    """Loads Gemma 2B model with 4-bit quantization and LoRA."""
    
    def __init__(self, config: Dict, hf_token: str = None):
        """
        Initialize model loader.
        
        Args:
            config: Configuration dictionary
            hf_token: HuggingFace token for gated models
        """
        self.config = config
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        self.model_name = config['model']['name']
        
    def load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer with quantization.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        print(f"Loading model: {self.model_name}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config['model']['load_in_4bit'],
            bnb_4bit_compute_dtype=getattr(torch, self.config['model']['bnb_4bit_compute_dtype']),
            bnb_4bit_quant_type=self.config['model']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=self.config['model']['bnb_4bit_use_double_quant']
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # Set padding token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.hf_token,
            trust_remote_code=True
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        print("Model loaded successfully with 4-bit quantization")
        
        return model, tokenizer
    
    def add_lora_adapters(self, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
        """
        Add LoRA adapters to model.
        
        Args:
            model: Base model
            
        Returns:
            Model with LoRA adapters
        """
        print("Adding LoRA adapters...")
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=self.config['lora']['task_type']
        )
        
        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        return model
    
    def load_for_training(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model and tokenizer ready for training.
        
        Returns:
            Tuple of (model with LoRA, tokenizer)
        """
        model, tokenizer = self.load_model_and_tokenizer()
        model = self.add_lora_adapters(model)
        return model, tokenizer
    
    def load_for_inference(self, checkpoint_path: str = None) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load model for inference.
        
        Args:
            checkpoint_path: Path to trained checkpoint (optional)
            
        Returns:
            Tuple of (model, tokenizer)
        """
        model, tokenizer = self.load_model_and_tokenizer()
        
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            model = self.add_lora_adapters(model)
            # Load adapter weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path)
        
        model.eval()
        return model, tokenizer


def load_model(config: Dict, hf_token: str = None, 
               for_training: bool = True, checkpoint_path: str = None):
    """
    Convenience function to load model.
    
    Args:
        config: Configuration dictionary
        hf_token: HuggingFace token
        for_training: Whether to load for training or inference
        checkpoint_path: Path to checkpoint for inference
        
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = GemmaModelLoader(config, hf_token)
    
    if for_training:
        return loader.load_for_training()
    else:
        return loader.load_for_inference(checkpoint_path)
