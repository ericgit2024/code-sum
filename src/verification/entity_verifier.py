"""
Entity verifier for comparing code and docstring entities.
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from .entity_extractor import CodeEntities, DocstringEntities, EntityExtractor


@dataclass
class EntityVerificationResult:
    """Result of entity verification."""
    precision: float
    recall: float
    f1_score: float
    hallucination_score: float
    
    # Detailed breakdown
    true_positives: Set[str]
    false_positives: Set[str]  # Hallucinated entities
    false_negatives: Set[str]  # Missing entities
    
    # Category-specific scores
    parameter_precision: float
    parameter_recall: float
    function_precision: float
    function_recall: float
    
    # Pass/fail
    passes_threshold: bool
    threshold: float


class EntityVerifier:
    """Verify entities between code and docstring."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize entity verifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.entity_extractor = EntityExtractor(config)
        
        # Get configuration
        verification_config = self.config.get('entity_verification', {})
        self.hallucination_threshold = verification_config.get('hallucination_threshold', 0.30)
        self.min_recall = verification_config.get('min_recall', 0.50)  # Require at least 50% of entities mentioned
        self.min_f1_score = verification_config.get('min_f1_score', 0.40)  # Require balanced precision/recall
        self.entity_weights = verification_config.get('entity_weights', {
            'function_names': 1.0,
            'parameter_names': 1.0,
            'called_functions': 0.7,
            'return_types': 0.7,
            'variables': 0.3
        })
        self.require_all_params = verification_config.get('require_all_params', True)
        self.allow_synonyms = verification_config.get('allow_synonyms', True)
    
    def verify(self, code: str, docstring: str) -> EntityVerificationResult:
        """
        Verify entities between code and docstring.
        
        Args:
            code: Python source code
            docstring: Generated docstring
            
        Returns:
            EntityVerificationResult with scores and details
        """
        # Extract entities
        code_entities = self.entity_extractor.extract_from_code(code)
        doc_entities = self.entity_extractor.extract_from_docstring(docstring)
        
        # Verify parameters (most critical)
        param_tp, param_fp, param_fn = self._compare_entities(
            set(code_entities.parameters),
            doc_entities.mentioned_parameters
        )
        
        # Verify called functions
        func_tp, func_fp, func_fn = self._compare_entities(
            code_entities.called_functions,
            doc_entities.mentioned_functions
        )
        
        # Combine all true positives, false positives, false negatives
        all_tp = param_tp | func_tp
        all_fp = param_fp | func_fp
        all_fn = param_fn | func_fn
        
        # Calculate overall metrics
        precision = self._calculate_precision(all_tp, all_fp)
        recall = self._calculate_recall(all_tp, all_fn)
        f1_score = self._calculate_f1(precision, recall)
        hallucination_score = 1.0 - precision
        
        # Calculate category-specific scores
        param_precision = self._calculate_precision(param_tp, param_fp)
        param_recall = self._calculate_recall(param_tp, param_fn)
        func_precision = self._calculate_precision(func_tp, func_fp)
        func_recall = self._calculate_recall(func_tp, func_fn)
        
        # Check if passes threshold (requires ALL criteria)
        passes_threshold = (
            hallucination_score <= self.hallucination_threshold and
            recall >= self.min_recall and
            f1_score >= self.min_f1_score
        )
        
        return EntityVerificationResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            hallucination_score=hallucination_score,
            true_positives=all_tp,
            false_positives=all_fp,
            false_negatives=all_fn,
            parameter_precision=param_precision,
            parameter_recall=param_recall,
            function_precision=func_precision,
            function_recall=func_recall,
            passes_threshold=passes_threshold,
            threshold=self.hallucination_threshold
        )
    
    def _compare_entities(self, code_entities: Set[str], 
                         doc_entities: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Compare two sets of entities.
        
        Args:
            code_entities: Entities from code
            doc_entities: Entities from docstring
            
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        true_positives = set()
        false_positives = set()
        false_negatives = set()
        
        # Normalize entities for comparison
        code_normalized = {self.entity_extractor.normalize_entity(e): e 
                          for e in code_entities}
        doc_normalized = {self.entity_extractor.normalize_entity(e): e 
                         for e in doc_entities}
        
        # Find true positives (mentioned and exists)
        for doc_norm, doc_orig in doc_normalized.items():
            if doc_norm in code_normalized:
                true_positives.add(doc_orig)
            elif self.allow_synonyms:
                # Check for synonyms
                found_synonym = False
                for code_norm, code_orig in code_normalized.items():
                    if self.entity_extractor.are_synonyms(doc_orig, code_orig):
                        true_positives.add(doc_orig)
                        found_synonym = True
                        break
                if not found_synonym:
                    false_positives.add(doc_orig)
            else:
                false_positives.add(doc_orig)
        
        # Find false negatives (exists but not mentioned)
        for code_norm, code_orig in code_normalized.items():
            if code_norm not in doc_normalized:
                # Check if any doc entity is a synonym
                found_synonym = False
                if self.allow_synonyms:
                    for doc_norm, doc_orig in doc_normalized.items():
                        if self.entity_extractor.are_synonyms(code_orig, doc_orig):
                            found_synonym = True
                            break
                if not found_synonym:
                    false_negatives.add(code_orig)
        
        return true_positives, false_positives, false_negatives
    
    def _calculate_precision(self, true_positives: Set[str], 
                            false_positives: Set[str]) -> float:
        """Calculate precision."""
        total = len(true_positives) + len(false_positives)
        if total == 0:
            return 1.0  # No entities mentioned = no hallucinations
        return len(true_positives) / total
    
    def _calculate_recall(self, true_positives: Set[str], 
                         false_negatives: Set[str]) -> float:
        """Calculate recall."""
        total = len(true_positives) + len(false_negatives)
        if total == 0:
            return 1.0  # No entities in code = nothing to recall
        return len(true_positives) / total
    
    def _calculate_f1(self, precision: float, recall: float) -> float:
        """Calculate F1 score."""
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def get_verification_summary(self, result: EntityVerificationResult) -> str:
        """
        Get human-readable summary of verification result.
        
        Args:
            result: EntityVerificationResult
            
        Returns:
            Formatted summary string
        """
        status = "✅ PASS" if result.passes_threshold else "❌ FAIL"
        
        summary = f"""
Entity Verification {status}
{'='*50}
Hallucination Score: {result.hallucination_score:.3f} (threshold: {result.threshold:.3f})
Precision: {result.precision:.3f} | Recall: {result.recall:.3f} | F1: {result.f1_score:.3f}

Parameter Accuracy:
  Precision: {result.parameter_precision:.3f} | Recall: {result.parameter_recall:.3f}

Function Accuracy:
  Precision: {result.function_precision:.3f} | Recall: {result.function_recall:.3f}

Details:
  ✓ Correct entities: {len(result.true_positives)}
  ✗ Hallucinated entities: {len(result.false_positives)}
  ⚠ Missing entities: {len(result.false_negatives)}
"""
        
        if result.false_positives:
            summary += f"\nHallucinated: {', '.join(sorted(result.false_positives))}"
        
        if result.false_negatives:
            summary += f"\nMissing: {', '.join(sorted(result.false_negatives))}"
        
        return summary.strip()
