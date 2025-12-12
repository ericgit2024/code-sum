"""
Dataset loader for CodeSearchNet Python dataset with sampling and quality filtering.
"""

import os
from datasets import load_dataset
from typing import Dict, List, Tuple
import random
from src.data.quality_filter import QualityFilter


class CodeSearchNetLoader:
    """Loads and prepares CodeSearchNet Python dataset."""
    
    def __init__(self, config: Dict):
        """
        Initialize dataset loader.
        
        Args:
            config: Configuration dictionary containing dataset parameters
        """
        self.config = config
        self.dataset_name = config['dataset']['name']
        self.language = config['dataset']['language']
        self.sample_size = config['dataset']['sample_size']
        self.cache_dir = config['dataset']['cache_dir']
        self.train_split = config['dataset']['train_split']
        self.val_split = config['dataset']['val_split']
        self.test_split = config['dataset']['test_split']
        
    def load_and_sample(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Load CodeSearchNet dataset and create sampled splits.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        print(f"Loading CodeXGLUE code-to-text dataset for {self.language}...")
        
        # Load CodeXGLUE dataset (code_x_glue_ct_code_to_text)
        # This dataset is well-maintained and doesn't use deprecated loading scripts
        # Available configs: 'go', 'java', 'javascript', 'php', 'python', 'ruby'
        dataset = load_dataset(
            "code_x_glue_ct_code_to_text",
            self.language,  # Use language directly, not "code_to_text-python"
            cache_dir=self.cache_dir
        )
        
        # Combine train and validation for sampling
        combined_data = []
        
        if 'train' in dataset:
            combined_data.extend(list(dataset['train']))
        if 'validation' in dataset:
            combined_data.extend(list(dataset['validation']))
            
        print(f"Total available samples: {len(combined_data)}")
        
        # Sample the required number of examples
        if len(combined_data) > self.sample_size:
            sampled_data = random.sample(combined_data, self.sample_size)
        else:
            sampled_data = combined_data
            print(f"Warning: Requested {self.sample_size} samples but only {len(combined_data)} available")
        
        # Shuffle the sampled data
        random.shuffle(sampled_data)
        
        # Prepare samples first (convert to standard format)
        prepared_data = self.prepare_dataset(sampled_data)
        
        # Apply quality filtering if enabled
        quality_filter_enabled = self.config.get('dataset', {}).get('quality_filter_enabled', True)
        if quality_filter_enabled:
            print(f"\nApplying quality filtering...")
            quality_filter = QualityFilter(self.config.get('quality_filter', {}))
            prepared_data, filter_stats = quality_filter.filter_dataset(prepared_data, verbose=True)
            
            # If we filtered out too many samples, warn user
            if filter_stats['retention_rate'] < 0.5:
                print(f"WARNING: Quality filter removed {filter_stats['removed_count']} samples!")
                print(f"Consider adjusting quality filter thresholds in config.yaml")
        else:
            print("\nQuality filtering disabled")
        
        # Create splits from filtered data
        train_size = int(len(prepared_data) * self.train_split)
        val_size = int(len(prepared_data) * self.val_split)
        
        train_data = prepared_data[:train_size]
        val_data = prepared_data[train_size:train_size + val_size]
        test_data = prepared_data[train_size + val_size:]
        
        print(f"\nFinal dataset splits:")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Validation: {len(val_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        
        return train_data, val_data, test_data
    
    def prepare_sample(self, sample: Dict) -> Dict:
        """
        Prepare a single sample for processing.
        
        Args:
            sample: Raw dataset sample
            
        Returns:
            Processed sample with code and docstring
        """
        # CodeXGLUE uses 'code' and 'docstring' fields
        return {
            'code': sample.get('code', sample.get('func_code_string', sample.get('whole_func_string', ''))),
            'docstring': sample.get('docstring', sample.get('func_documentation_string', '')),
            'func_name': sample.get('func_name', ''),
            'repo': sample.get('repo', ''),
            'path': sample.get('path', '')
        }
    
    def prepare_dataset(self, data: List[Dict]) -> List[Dict]:
        """
        Prepare entire dataset split.
        
        Args:
            data: List of raw samples
            
        Returns:
            List of processed samples
        """
        return [self.prepare_sample(sample) for sample in data]


def load_codesearchnet_dataset(config: Dict) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convenience function to load and prepare CodeSearchNet dataset with quality filtering.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_data, val_data, test_data) - already prepared and filtered
    """
    loader = CodeSearchNetLoader(config)
    train_data, val_data, test_data = loader.load_and_sample()
    
    return train_data, val_data, test_data

