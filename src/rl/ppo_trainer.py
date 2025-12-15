"""
PPO Trainer for RL-based docstring generation.

Uses Proximal Policy Optimization (PPO) with custom execution-based rewards
to fine-tune the model for better docstring quality.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from src.rl.reward_function import RewardFunction
from src.data.preprocessor import DataPreprocessor


class DocstringPPOTrainer:
    """PPO trainer for docstring generation with execution-based rewards."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 config: Dict, preprocessor: DataPreprocessor):
        """
        Initialize PPO trainer.
        
        Args:
            model: Base language model
            tokenizer: Tokenizer
            config: Configuration dictionary
            preprocessor: Data preprocessor for formatting prompts
        """
        self.tokenizer = tokenizer
        self.config = config
        self.preprocessor = preprocessor
        
        # Initialize reward function
        self.reward_fn = RewardFunction(config)
        
        # PPO configuration
        rl_config = config.get('rl', {})
        self.ppo_config = PPOConfig(
            model_name=config['model']['name'],
            learning_rate=rl_config.get('learning_rate', 1e-5),
            batch_size=rl_config.get('batch_size', 4),
            mini_batch_size=rl_config.get('mini_batch_size', 1),
            gradient_accumulation_steps=rl_config.get('gradient_accumulation_steps', 4),
            ppo_epochs=rl_config.get('ppo_epochs', 4),
            max_grad_norm=rl_config.get('max_grad_norm', 1.0),
            optimize_cuda_cache=True,
            early_stopping=False,
            target_kl=rl_config.get('target_kl', 0.01),
            seed=42,
        )
        
        # Wrap model with value head for PPO
        print("Wrapping model with value head...")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        
        # Initialize PPO trainer
        print("Initializing PPO trainer...")
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=tokenizer,
        )
        
        # Training parameters
        self.kl_penalty = rl_config.get('kl_penalty', 0.1)
        self.clip_range = rl_config.get('clip_range', 0.2)
        
        # Generation parameters
        self.gen_kwargs = {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        
        print(f"PPO Trainer initialized with:")
        print(f"  Learning rate: {self.ppo_config.learning_rate}")
        print(f"  Batch size: {self.ppo_config.batch_size}")
        print(f"  PPO epochs: {self.ppo_config.ppo_epochs}")
        print(f"  KL penalty: {self.kl_penalty}")
    
    def prepare_prompts(self, code_batch: List[str]) -> List[str]:
        """
        Prepare prompts for code samples.
        
        Args:
            code_batch: List of code samples
            
        Returns:
            List of formatted prompts
        """
        prompts = []
        for code in code_batch:
            # Extract structures
            structures = self.preprocessor.extract_structures(code)
            # Format prompt (without RAG context)
            prompt = self.preprocessor.format_prompt(code, structures, rag_context="")
            prompts.append(prompt)
        
        return prompts
    
    def generate_docstrings(self, prompts: List[str]) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate docstrings for prompts.
        
        Args:
            prompts: List of formatted prompts
            
        Returns:
            Tuple of (generated_docstrings, response_tensors)
        """
        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.pretrained_model.device)
        
        # Generate
        response_tensors = []
        generated_texts = []
        
        with torch.no_grad():
            for i in range(len(prompts)):
                input_ids = inputs['input_ids'][i:i+1]
                attention_mask = inputs['attention_mask'][i:i+1]
                
                # Generate response
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **self.gen_kwargs,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                
                # Extract generated tokens (excluding prompt)
                response = outputs.sequences[0][input_ids.shape[1]:]
                response_tensors.append(response)
                
                # Decode
                generated_text = self.tokenizer.decode(response, skip_special_tokens=True)
                generated_texts.append(self._clean_docstring(generated_text))
        
        return generated_texts, response_tensors
    
    def _clean_docstring(self, text: str) -> str:
        """Clean generated docstring."""
        # Remove common markers
        for marker in ["Docstring:", "Summary:", "Output:"]:
            if marker in text:
                text = text.split(marker)[-1].strip()
        
        # Remove triple quotes
        text = text.replace('"""', '').replace("'''", '')
        
        # Take first few sentences (avoid overly long outputs)
        sentences = text.split('.')
        if len(sentences) > 4:
            text = '. '.join(sentences[:4]) + '.'
        
        return text.strip()
    
    def compute_rewards(self, code_batch: List[str], docstring_batch: List[str],
                       reference_batch: List[str] = None) -> List[float]:
        """
        Compute rewards for generated docstrings.
        
        Args:
            code_batch: List of code samples
            docstring_batch: List of generated docstrings
            reference_batch: Optional list of reference docstrings
            
        Returns:
            List of reward scores
        """
        rewards = []
        
        for i, (code, docstring) in enumerate(zip(code_batch, docstring_batch)):
            reference = reference_batch[i] if reference_batch else None
            reward, _ = self.reward_fn.compute_reward(code, docstring, reference)
            rewards.append(reward)
        
        return rewards
    
    def train_step(self, code_batch: List[str], reference_batch: List[str]) -> Dict:
        """
        Execute one PPO training step.
        
        Args:
            code_batch: List of code samples
            reference_batch: List of reference docstrings
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare prompts
        prompts = self.prepare_prompts(code_batch)
        
        # Tokenize prompts for PPO
        query_tensors = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")[0]
            query_tensors.append(tokens)
        
        # Generate docstrings
        generated_docstrings, response_tensors = self.generate_docstrings(prompts)
        
        # Compute rewards
        reward_scores = self.compute_rewards(code_batch, generated_docstrings, reference_batch)
        rewards = [torch.tensor(r) for r in reward_scores]
        
        # PPO step
        stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Compile metrics
        metrics = {
            'mean_reward': np.mean(reward_scores),
            'std_reward': np.std(reward_scores),
            'min_reward': np.min(reward_scores),
            'max_reward': np.max(reward_scores),
            'ppo_loss': stats.get('ppo/loss/total', 0),
            'kl_divergence': stats.get('objective/kl', 0),
        }
        
        return metrics
    
    def save_checkpoint(self, output_dir: str):
        """
        Save model checkpoint.
        
        Args:
            output_dir: Directory to save checkpoint
        """
        print(f"Saving checkpoint to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Checkpoint saved!")
