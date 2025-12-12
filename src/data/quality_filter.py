"""
Quality filtering for code-summary pairs.
Filters out low-quality samples to improve training data.
"""

import re
from typing import Dict, List, Tuple


class QualityFilter:
    """Filters code-summary pairs based on quality heuristics."""
    
    def __init__(self, config: Dict = None):
        """
        Initialize quality filter.
        
        Args:
            config: Optional configuration for thresholds
        """
        self.config = config or {}
        
        # Default thresholds (can be overridden in config)
        self.min_code_length = self.config.get('min_code_length', 20)
        self.max_code_length = self.config.get('max_code_length', 2000)
        self.min_summary_length = self.config.get('min_summary_length', 10)
        self.max_summary_length = self.config.get('max_summary_length', 500)
        self.min_summary_words = self.config.get('min_summary_words', 3)
        self.max_summary_words = self.config.get('max_summary_words', 100)
        self.min_code_lines = self.config.get('min_code_lines', 2)
        self.max_code_lines = self.config.get('max_code_lines', 100)
    
    def is_high_quality(self, sample: Dict) -> Tuple[bool, str]:
        """
        Check if a code-summary pair is high quality.
        
        Args:
            sample: Sample with 'code' and 'docstring' or 'summary'
            
        Returns:
            Tuple of (is_quality, reason)
        """
        code = sample.get('code', '')
        summary = sample.get('docstring', sample.get('summary', ''))
        
        # Check 1: Non-empty
        if not code or not summary:
            return False, "Empty code or summary"
        
        # Check 2: Code length
        if len(code) < self.min_code_length:
            return False, f"Code too short ({len(code)} chars)"
        if len(code) > self.max_code_length:
            return False, f"Code too long ({len(code)} chars)"
        
        # Check 3: Summary length
        if len(summary) < self.min_summary_length:
            return False, f"Summary too short ({len(summary)} chars)"
        if len(summary) > self.max_summary_length:
            return False, f"Summary too long ({len(summary)} chars)"
        
        # Check 4: Summary word count
        summary_words = summary.split()
        if len(summary_words) < self.min_summary_words:
            return False, f"Summary too few words ({len(summary_words)})"
        if len(summary_words) > self.max_summary_words:
            return False, f"Summary too many words ({len(summary_words)})"
        
        # Check 5: Code line count
        code_lines = [line for line in code.split('\n') if line.strip()]
        if len(code_lines) < self.min_code_lines:
            return False, f"Code too few lines ({len(code_lines)})"
        if len(code_lines) > self.max_code_lines:
            return False, f"Code too many lines ({len(code_lines)})"
        
        # Check 6: Summary is not just code
        if self._is_code_like(summary):
            return False, "Summary looks like code"
        
        # Check 7: Summary is not auto-generated placeholder
        if self._is_placeholder(summary):
            return False, "Summary is placeholder"
        
        # Check 8: Summary is not just function name
        func_name = sample.get('func_name', '')
        if func_name and summary.strip().lower() == func_name.lower():
            return False, "Summary is just function name"
        
        # Check 9: Code is parseable Python
        if not self._is_valid_python(code):
            return False, "Code is not valid Python"
        
        # Check 10: Summary has meaningful content
        if not self._has_meaningful_content(summary):
            return False, "Summary lacks meaningful content"
        
        # Check 11: Code-summary relevance (basic check)
        if not self._is_relevant(code, summary):
            return False, "Code and summary seem unrelated"
        
        return True, "High quality"
    
    def _is_code_like(self, text: str) -> bool:
        """Check if text looks like code rather than natural language."""
        # Check for common code patterns
        code_patterns = [
            r'def\s+\w+\s*\(',  # Function definition
            r'class\s+\w+',     # Class definition
            r'import\s+\w+',    # Import statement
            r'return\s+\w+',    # Return statement
            r'if\s+\w+\s*[=<>]', # If condition
            r'for\s+\w+\s+in',  # For loop
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for high ratio of special characters
        special_chars = sum(1 for c in text if c in '{}[]();=<>+-*/')
        if len(text) > 0 and special_chars / len(text) > 0.2:
            return True
        
        return False
    
    def _is_placeholder(self, text: str) -> bool:
        """Check if text is an auto-generated placeholder."""
        text_lower = text.lower().strip()
        
        placeholders = [
            'todo',
            'fixme',
            'placeholder',
            'description here',
            'add description',
            'no description',
            'none',
            '...',
            'tbd',
            'to be determined',
        ]
        
        return any(p in text_lower for p in placeholders)
    
    def _is_valid_python(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        import ast
        try:
            ast.parse(code)
            return True
        except:
            return False
    
    def _has_meaningful_content(self, text: str) -> bool:
        """Check if text has meaningful content (not just stopwords)."""
        # Remove common stopwords
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                     'this', 'that', 'these', 'those', 'to', 'of', 'in', 'on',
                     'at', 'for', 'with', 'by', 'from', 'as'}
        
        words = text.lower().split()
        meaningful_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # At least 2 meaningful words
        return len(meaningful_words) >= 2
    
    def _is_relevant(self, code: str, summary: str) -> bool:
        """
        Basic relevance check between code and summary.
        Checks if summary mentions key elements from code.
        """
        # Extract function name from code
        import ast
        try:
            tree = ast.parse(code)
            func_names = [node.name for node in ast.walk(tree) 
                         if isinstance(node, ast.FunctionDef)]
            
            if not func_names:
                return True  # No function, can't check
            
            # Check if summary is completely generic (no specific terms)
            summary_lower = summary.lower()
            
            # Generic summaries that don't mention anything specific
            generic_phrases = [
                'this function',
                'this method',
                'the function',
                'the method',
            ]
            
            # If summary is ONLY generic phrases, it's low quality
            if any(summary_lower.strip().startswith(phrase) for phrase in generic_phrases):
                # Check if it has ANY specific content after the generic phrase
                words = summary_lower.split()
                if len(words) < 5:  # Too short and generic
                    return False
            
            return True
            
        except:
            return True  # Can't parse, give benefit of doubt
    
    def filter_dataset(self, dataset: List[Dict], 
                      verbose: bool = True) -> Tuple[List[Dict], Dict]:
        """
        Filter dataset to keep only high-quality samples.
        
        Args:
            dataset: List of samples
            verbose: Print filtering statistics
            
        Returns:
            Tuple of (filtered_dataset, statistics)
        """
        filtered = []
        rejection_reasons = {}
        
        for sample in dataset:
            is_quality, reason = self.is_high_quality(sample)
            
            if is_quality:
                filtered.append(sample)
            else:
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        
        # Calculate statistics
        stats = {
            'original_count': len(dataset),
            'filtered_count': len(filtered),
            'removed_count': len(dataset) - len(filtered),
            'retention_rate': len(filtered) / len(dataset) if dataset else 0,
            'rejection_reasons': rejection_reasons
        }
        
        if verbose:
            self._print_statistics(stats)
        
        return filtered, stats
    
    def _print_statistics(self, stats: Dict):
        """Print filtering statistics."""
        print("\n" + "="*60)
        print("QUALITY FILTERING RESULTS")
        print("="*60)
        print(f"Original samples: {stats['original_count']}")
        print(f"High-quality samples: {stats['filtered_count']}")
        print(f"Removed samples: {stats['removed_count']}")
        print(f"Retention rate: {stats['retention_rate']:.1%}")
        
        if stats['rejection_reasons']:
            print("\nRejection reasons:")
            for reason, count in sorted(stats['rejection_reasons'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count} samples")
        print("="*60 + "\n")


def filter_quality_samples(dataset: List[Dict], 
                          config: Dict = None,
                          verbose: bool = True) -> List[Dict]:
    """
    Convenience function to filter dataset.
    
    Args:
        dataset: List of samples
        config: Optional configuration
        verbose: Print statistics
        
    Returns:
        Filtered dataset
    """
    filter = QualityFilter(config)
    filtered, stats = filter.filter_dataset(dataset, verbose)
    return filtered
