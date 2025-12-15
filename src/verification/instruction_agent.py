"""
Instruction agent for generating refinement feedback based on entity verification.
"""

from typing import Dict, List
from .entity_verifier import EntityVerificationResult


class InstructionAgent:
    """Generate specific instructions for model refinement based on entity verification."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize instruction agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def generate_instructions(self, result: EntityVerificationResult, 
                            code: str, docstring: str) -> str:
        """
        Generate specific instructions for refinement based on verification result.
        
        Args:
            result: EntityVerificationResult
            code: Original code
            docstring: Current docstring
            
        Returns:
            Detailed instruction string for model refinement
        """
        if result.passes_threshold:
            return ""  # No instructions needed if passing
        
        instructions = []
        
        # Header
        instructions.append(f"⚠️ Entity Verification Failed (Hallucination Score: {result.hallucination_score:.2f} > {result.threshold:.2f})")
        instructions.append("")
        
        # Hallucinated entities (false positives)
        if result.false_positives:
            instructions.append("❌ Remove these hallucinated entities (they don't exist in the code):")
            for entity in sorted(result.false_positives):
                instructions.append(f"  - '{entity}'")
            instructions.append("")
        
        # Missing entities (false negatives)
        if result.false_negatives:
            instructions.append("➕ Include these missing entities in the description:")
            for entity in sorted(result.false_negatives):
                instructions.append(f"  - '{entity}'")
            instructions.append("")
        
        # Specific guidance based on entity types
        if result.parameter_recall < 0.8:
            instructions.append("📝 Ensure all function parameters are mentioned and described.")
        
        if result.parameter_precision < 0.8:
            instructions.append("🔍 Check that all mentioned parameters actually exist in the function signature.")
        
        if result.function_precision < 0.7:
            instructions.append("⚠️ Verify that mentioned function calls are actually present in the code.")
        
        instructions.append("")
        instructions.append("📋 Action Required:")
        instructions.append("Rewrite the docstring to:")
        
        action_items = []
        if result.false_positives:
            action_items.append(f"1. Remove references to: {', '.join(sorted(result.false_positives))}")
        if result.false_negatives:
            action_items.append(f"2. Add descriptions for: {', '.join(sorted(result.false_negatives))}")
        action_items.append(f"{len(action_items) + 1}. Ensure accuracy - only describe what the code actually does")
        
        instructions.extend(action_items)
        
        return "\n".join(instructions)
    
    def generate_critique_enhancement(self, result: EntityVerificationResult) -> str:
        """
        Generate enhancement to add to reflective agent critique.
        
        Args:
            result: EntityVerificationResult
            
        Returns:
            Critique enhancement text
        """
        if result.passes_threshold:
            return f"✓ Entity verification passed (hallucination score: {result.hallucination_score:.2f})"
        
        critique_parts = []
        critique_parts.append(f"Entity Verification: FAILED (hallucination score: {result.hallucination_score:.2f} > threshold: {result.threshold:.2f})")
        
        if result.false_positives:
            critique_parts.append(f"Hallucinated entities: {', '.join(sorted(result.false_positives))}")
        
        if result.false_negatives:
            critique_parts.append(f"Missing entities: {', '.join(sorted(result.false_negatives))}")
        
        return " | ".join(critique_parts)
    
    def create_refinement_prompt(self, code: str, docstring: str, 
                                result: EntityVerificationResult) -> str:
        """
        Create a refinement prompt for the model with entity-specific feedback.
        
        Args:
            code: Original code
            docstring: Current docstring
            result: EntityVerificationResult
            
        Returns:
            Formatted refinement prompt
        """
        instructions = self.generate_instructions(result, code, docstring)
        
        prompt = f"""The current docstring has entity verification issues. Please revise it.

Code:
{code}

Current Docstring:
{docstring}

{instructions}

Write a revised docstring that addresses all the issues above. Focus on accuracy and completeness.

Revised Docstring:"""
        
        return prompt
    
    def get_feedback_summary(self, result: EntityVerificationResult) -> Dict[str, any]:
        """
        Get structured feedback summary for logging/analysis.
        
        Args:
            result: EntityVerificationResult
            
        Returns:
            Dictionary with feedback details
        """
        return {
            'passes': result.passes_threshold,
            'hallucination_score': result.hallucination_score,
            'threshold': result.threshold,
            'precision': result.precision,
            'recall': result.recall,
            'f1_score': result.f1_score,
            'hallucinated_count': len(result.false_positives),
            'missing_count': len(result.false_negatives),
            'hallucinated_entities': list(result.false_positives),
            'missing_entities': list(result.false_negatives),
            'correct_entities': list(result.true_positives),
            'parameter_precision': result.parameter_precision,
            'parameter_recall': result.parameter_recall,
            'function_precision': result.function_precision,
            'function_recall': result.function_recall
        }
