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
                max_new_tokens=400,  # Increased from 256 to prevent truncation
                min_new_tokens=20,   # Ensure minimum output length
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
        
        # Extract summary (after "Summary:" or "Docstring:" marker)
        summary = generated_text.strip()
        for marker in ["Summary:", "Docstring:", "Output:"]:
            if marker in summary:
                summary = summary.split(marker)[-1].strip()
                break
        
        # Clean the summary to remove code artifacts
        summary = self._clean_generated_text(summary)
        
        return summary
    
    def _clean_generated_text(self, text: str) -> str:
        """
        Clean generated text by removing code artifacts.
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text
        """
        # Debug: print raw output to help diagnose issues
        if text.strip():
            print(f"[DEBUG] Raw LLM output: {repr(text[:200])}")
        
        # Extract text from within triple quotes if present
        if '"""' in text or "'''" in text:
            # Try to extract content between triple quotes
            for quote in ['"""', "'''"]:
                if text.count(quote) >= 2:
                    parts = text.split(quote)
                    if len(parts) >= 3:
                        # Take the content between first pair of triple quotes
                        text = parts[1].strip()
                        print(f"[DEBUG] Extracted from quotes: {repr(text[:100])}")
                        break
        
        # Extract text from code blocks (markdown format)
        if '```' in text:
            # Try to extract content between code blocks
            parts = text.split('```')
            if len(parts) >= 3:
                # Content is between first pair of ```
                # Skip language identifier if present (e.g., ```python)
                content = parts[1].strip()
                if '\n' in content:
                    lines = content.split('\n')
                    # Skip first line if it's a language identifier
                    if lines[0].strip() in ['python', 'py', 'text', '']:
                        content = '\n'.join(lines[1:]).strip()
                text = content
                print(f"[DEBUG] Extracted from code block: {repr(text[:100])}")
            else:
                # Just remove the code block markers
                text = text.replace('```', '').strip()
        
        # Remove lines that look like code (but be less aggressive)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Only skip lines that START with code keywords (not just contain them)
            if line.startswith(('def ', 'class ', 'import ', 'from ')):
                continue
            # Skip standalone triple quotes (already extracted content above)
            if line in ['"""', "'''"]:
                continue
            cleaned_lines.append(line)
        
        # Join lines
        result = ' '.join(cleaned_lines)
        
        # Remove multiple spaces
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        # Deduplicate repetitive sentences
        result = self._deduplicate_text(result)
        
        print(f"[DEBUG] Final cleaned: {repr(result[:100])}")
        
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
