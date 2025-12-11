"""
Training script for Gemma 2B with LoRA on code summarization.
"""

import os
import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from typing import Dict, List
import yaml


class CodeSummarizationTrainer:
    """Trainer for code summarization model."""
    
    def __init__(self, model, tokenizer, config: Dict):
        """
        Initialize trainer.
        
        Args:
            model: Model with LoRA adapters
            tokenizer: Tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_seq_length = config['training']['max_seq_length']
        
    def prepare_training_data(self, preprocessed_data: List[Dict]) -> Dataset:
        """
        Prepare data for training.
        
        Args:
            preprocessed_data: List of preprocessed samples
            
        Returns:
            HuggingFace Dataset
        """
        # Format data for training
        formatted_data = []
        
        for sample in preprocessed_data:
            # Create input-output pair
            text = f"{sample['prompt']}\n\nSummary: {sample['target']}"
            formatted_data.append({'text': text})
        
        # Create dataset
        dataset = Dataset.from_list(formatted_data)
        
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize examples for training."""
        # Tokenize
        tokenized = self.tokenizer(
            examples['text'],
            truncation=True,
            max_length=self.max_seq_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Set labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].clone()
        
        return tokenized
    
    def train(self, train_data: List[Dict], val_data: List[Dict]):
        """
        Train the model.
        
        Args:
            train_data: Preprocessed training data
            val_data: Preprocessed validation data
        """
        print("Preparing training datasets...")
        
        # Prepare datasets
        train_dataset = self.prepare_training_data(train_data)
        val_dataset = self.prepare_training_data(val_data)
        
        # Tokenize
        train_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        val_dataset = val_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['training']['output_dir'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_eval_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            learning_rate=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            warmup_steps=self.config['training']['warmup_steps'],
            logging_steps=self.config['training']['logging_steps'],
            save_steps=self.config['training']['save_steps'],
            eval_steps=self.config['training']['eval_steps'],
            save_total_limit=self.config['training']['save_total_limit'],
            fp16=self.config['training']['fp16'],
            optim=self.config['training']['optim'],
            max_grad_norm=self.config['training']['max_grad_norm'],
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["tensorboard"],
            logging_dir=os.path.join(self.config['training']['output_dir'], "logs")
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator
        )
        
        # Train
        print("Starting training...")
        trainer.train()
        
        # Save final model
        final_output_dir = os.path.join(self.config['training']['output_dir'], "final_model")
        trainer.save_model(final_output_dir)
        self.tokenizer.save_pretrained(final_output_dir)
        
        print(f"Training complete! Model saved to {final_output_dir}")


def train_model(model, tokenizer, config: Dict, train_data: List[Dict], val_data: List[Dict]):
    """
    Convenience function to train model.
    
    Args:
        model: Model with LoRA
        tokenizer: Tokenizer
        config: Configuration
        train_data: Training data
        val_data: Validation data
    """
    trainer = CodeSummarizationTrainer(model, tokenizer, config)
    trainer.train(train_data, val_data)
