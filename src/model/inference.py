"""
Inference pipeline for generating summaries with RAG and reflective agent.
"""

import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.rag.rag_system import RAGSystem
from src.data.preprocessor import DataPreprocessor
from src.agent.reflective_agent import ReflectiveAgent


class InferencePipeline:
    """End-to-end inference pipeline."""
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer,
                 rag_system: RAGSystem, preprocessor: DataPreprocessor,
                 reflective_agent: ReflectiveAgent, config: Dict):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained model
            tokenizer: Tokenizer
            rag_system: RAG system
            preprocessor: Data preprocessor
            reflective_agent: Reflective agent
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.rag_system = rag_system
        self.preprocessor = preprocessor
        self.reflective_agent = reflective_agent
        self.config = config
        
    def generate_initial_summary(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate initial summary from prompt.
        
        Args:
            prompt: Formatted prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated summary
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract summary (after "Summary:" marker)
        if "Summary:" in generated_text:
            summary = generated_text.split("Summary:")[-1].strip()
        else:
            summary = generated_text.strip()
        
        return summary
    
    def predict_single(self, code: str, use_reflective_agent: bool = True) -> Dict:
        """
        Generate summary for a single code sample.
        
        Args:
            code: Python source code
            use_reflective_agent: Whether to use reflective agent
            
        Returns:
            Dictionary with summary and metadata
        """
        # Retrieve similar examples
        retrieved = self.rag_system.retrieve(code)
        rag_context = self.rag_system.format_rag_context(retrieved)
        
        # Extract structures
        structures = self.preprocessor.extract_structures(code)
        
        # Format prompt
        prompt = self.preprocessor.format_prompt(code, structures, rag_context)
        
        # Generate initial summary
        initial_summary = self.generate_initial_summary(prompt)
        
        # Apply reflective agent if enabled
        if use_reflective_agent:
            final_summary, iterations = self.reflective_agent.iterative_refinement(
                code, initial_summary
            )
        else:
            final_summary = initial_summary
            iterations = 0
        
        return {
            'code': code,
            'initial_summary': initial_summary,
            'final_summary': final_summary,
            'iterations': iterations,
            'structures': structures,
            'rag_examples': len(retrieved)
        }
    
    def predict_batch(self, test_data: List[Dict], 
                     use_reflective_agent: bool = True) -> List[Dict]:
        """
        Generate summaries for a batch of samples.
        
        Args:
            test_data: List of test samples with 'code'
            use_reflective_agent: Whether to use reflective agent
            
        Returns:
            List of predictions
        """
        predictions = []
        
        print(f"Generating summaries for {len(test_data)} samples...")
        
        for i, sample in enumerate(test_data):
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_data)} samples")
            
            prediction = self.predict_single(
                sample['code'],
                use_reflective_agent=use_reflective_agent
            )
            
            # Add reference if available
            if 'docstring' in sample:
                prediction['reference'] = sample['docstring']
            
            predictions.append(prediction)
        
        print(f"Summary generation complete: {len(predictions)} samples")
        
        return predictions
