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
                max_new_tokens=256,  # Increased for better quality summaries
                temperature=0.7,
                do_sample=True,  # Enable sampling for diverse, higher-quality outputs
                top_p=0.9,  # Nucleus sampling for better quality
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract summary (after "Summary:" or "Docstring:" marker)
        summary = generated_text.strip()
        for marker in ["Summary:", "Docstring:", "Output:"]:
            if marker in summary:
                summary = summary.split(marker)[-1].strip()
                break
        
        # Clean the summary to remove code artifacts (minimal cleaning)
        summary = self._clean_generated_text(summary)
        
        return summary
    

    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean generated text by removing code artifacts (minimal processing).
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Extract text from within triple quotes if present
        if '"""' in text or "'''" in text:
            for quote in ['"""', "'''"]:
                if text.count(quote) >= 2:
                    parts = text.split(quote)
                    if len(parts) >= 3:
                        text = parts[1].strip()
                        break
        
        # If there's a code block, take the text BEFORE it (the summary), not the code itself
        if '```' in text:
            parts = text.split('```')
            if len(parts) >= 2:
                text = parts[0].strip()
            else:
                text = text.replace('```', '').strip()
        
        # Remove only obvious code lines (def, class, import)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Only skip lines that are clearly code definitions
            if line.startswith(('def ', 'class ', 'import ', 'from ')):
                continue
            if line in ['"""', "'''"]:
                continue
            cleaned_lines.append(line)
        
        # Join lines into a single paragraph
        result = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        return result.strip()
    
    def _deduplicate_text(self, text: str) -> str:
        """
        Remove repetitive sentences from text.
        
        Args:
            text: Text that may contain repetition
            
        Returns:
            Deduplicated text
        """
        # Split into sentences (simple approach)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        if not sentences:
            return text
        
        # Keep only unique sentences in order
        seen = set()
        unique_sentences = []
        
        for sentence in sentences:
            # Normalize for comparison (lowercase, no extra spaces)
            normalized = ' '.join(sentence.lower().split())
            if normalized not in seen:
                seen.add(normalized)
                unique_sentences.append(sentence)
        
        # Rejoin with periods
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        
        return result
    
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
            final_summary, iterations, metadata = self.reflective_agent.iterative_refinement(
                code, initial_summary
            )
            
            # Calculate improvement if scoring is enabled
            improvement = None
            if metadata.get('scores_history'):
                initial_score = metadata['scores_history'][0]['weighted_score'] if metadata['scores_history'] else 0
                final_score = metadata.get('final_score', 0)
                improvement = final_score - initial_score
        else:
            final_summary = initial_summary
            iterations = 0
            metadata = {}
            improvement = None
        
        result = {
            'code': code,
            'initial_summary': initial_summary,
            'final_summary': final_summary,
            'iterations': iterations,
            'structures': structures,
            'rag_examples': len(retrieved)
        }
        
        # Add scoring metadata if available
        if metadata:
            result['scores_history'] = metadata.get('scores_history', [])
            result['final_score'] = metadata.get('final_score')
            result['complexity'] = metadata.get('complexity', {})
            result['stop_reason'] = metadata.get('stop_reason')
            result['improvement'] = improvement
        
        return result
    
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
        import time
        
        predictions = []
        total_samples = len(test_data)
        
        print(f"Generating summaries for {total_samples} samples...")
        print(f"Reflective agent: {'ENABLED' if use_reflective_agent else 'DISABLED'}")
        
        start_time = time.time()
        
        for i, sample in enumerate(test_data):
            sample_start = time.time()
            
            prediction = self.predict_single(
                sample['code'],
                use_reflective_agent=use_reflective_agent
            )
            
            # Add reference if available
            if 'docstring' in sample:
                prediction['reference'] = sample['docstring']
            
            predictions.append(prediction)
            
            # Progress reporting every 10 samples
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                samples_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = total_samples - (i + 1)
                eta_seconds = remaining / samples_per_sec if samples_per_sec > 0 else 0
                eta_minutes = eta_seconds / 60
                
                print(f"Processed {i + 1}/{total_samples} samples | "
                      f"{samples_per_sec:.2f} samples/sec | "
                      f"ETA: {eta_minutes:.1f} min")
        
        total_time = time.time() - start_time
        avg_time_per_sample = total_time / total_samples if total_samples > 0 else 0
        
        print(f"\nSummary generation complete!")
        print(f"Total samples: {len(predictions)}")
        print(f"Total time: {total_time / 60:.2f} minutes")
        print(f"Average time per sample: {avg_time_per_sample:.2f} seconds")
        
        return predictions
