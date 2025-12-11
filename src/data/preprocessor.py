"""
Data preprocessing pipeline that combines structure extraction and formatting.
"""

from typing import Dict, List
from src.structure.ast_extractor import ASTExtractor
from src.structure.cfg_extractor import CFGExtractor
from src.structure.pdg_extractor import PDGExtractor


class DataPreprocessor:
    """Preprocesses code samples with structure extraction."""
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize extractors
        self.ast_extractor = ASTExtractor(
            max_depth=config['structure']['max_ast_depth']
        )
        self.cfg_extractor = CFGExtractor(
            max_nodes=config['structure']['max_cfg_nodes']
        )
        self.pdg_extractor = PDGExtractor(
            max_nodes=config['structure']['max_pdg_nodes']
        )
        
    def extract_structures(self, code: str) -> Dict[str, str]:
        """
        Extract all structural representations.
        
        Args:
            code: Python source code
            
        Returns:
            Dictionary with AST, CFG, and PDG encodings
        """
        structures = {}
        
        if self.config['structure']['extract_ast']:
            structures['ast'] = self.ast_extractor.extract_and_encode(code)
        else:
            structures['ast'] = ""
        
        if self.config['structure']['extract_cfg']:
            structures['cfg'] = self.cfg_extractor.extract_and_encode(code)
        else:
            structures['cfg'] = ""
        
        if self.config['structure']['extract_pdg']:
            structures['pdg'] = self.pdg_extractor.extract_and_encode(code)
        else:
            structures['pdg'] = ""
        
        return structures
    
    def format_prompt(self, code: str, structures: Dict[str, str], 
                     rag_context: str) -> str:
        """
        Format complete prompt for model.
        
        Args:
            code: Python source code
            structures: Extracted structures
            rag_context: RAG retrieval context
            
        Returns:
            Formatted prompt string
        """
        instruction_template = self.config['prompts']['instruction_template']
        
        prompt = instruction_template.format(
            code=code,
            ast=structures.get('ast', ''),
            cfg=structures.get('cfg', ''),
            pdg=structures.get('pdg', ''),
            rag_context=rag_context
        )
        
        return prompt
    
    def preprocess_sample(self, sample: Dict, rag_context: str = "") -> Dict:
        """
        Preprocess a single sample.
        
        Args:
            sample: Sample with 'code' and 'docstring'
            rag_context: RAG context (optional)
            
        Returns:
            Preprocessed sample with prompt and target
        """
        code = sample['code']
        
        # Extract structures
        structures = self.extract_structures(code)
        
        # Format prompt
        prompt = self.format_prompt(code, structures, rag_context)
        
        return {
            'code': code,
            'prompt': prompt,
            'target': sample['docstring'],
            'structures': structures
        }
    
    def preprocess_dataset(self, dataset: List[Dict], 
                          rag_system=None) -> List[Dict]:
        """
        Preprocess entire dataset.
        
        Args:
            dataset: List of samples
            rag_system: RAG system for retrieval (optional)
            
        Returns:
            List of preprocessed samples
        """
        preprocessed = []
        
        print(f"Preprocessing {len(dataset)} samples...")
        
        for i, sample in enumerate(dataset):
            if i % 100 == 0:
                print(f"Processed {i}/{len(dataset)} samples")
            
            # Get RAG context if available
            rag_context = ""
            if rag_system:
                retrieved = rag_system.retrieve(sample['code'])
                rag_context = rag_system.format_rag_context(retrieved)
            
            # Preprocess sample
            preprocessed_sample = self.preprocess_sample(sample, rag_context)
            preprocessed.append(preprocessed_sample)
        
        print(f"Preprocessing complete: {len(preprocessed)} samples")
        
        return preprocessed
