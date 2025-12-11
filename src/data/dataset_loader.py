"""
Dataset loader for CodeSearchNet Python dataset with sampling.
"""

import os
from datasets import load_dataset
from typing import Dict, List, Tuple
import random


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
        print(f"Loading {self.dataset_name} dataset for {self.language}...")
        
        # Load the dataset
        dataset = load_dataset(
            self.dataset_name,
            self.language,
            cache_dir=self.cache_dir,
            trust_remote_code=True
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
        
        # Create splits
        train_size = int(len(sampled_data) * self.train_split)
        val_size = int(len(sampled_data) * self.val_split)
        
        train_data = sampled_data[:train_size]
        val_data = sampled_data[train_size:train_size + val_size]
        test_data = sampled_data[train_size + val_size:]
        
        print(f"Dataset splits created:")
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
        return {
            'code': sample.get('func_code_string', sample.get('whole_func_string', '')),
            'docstring': sample.get('func_documentation_string', ''),
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
    Convenience function to load and prepare CodeSearchNet dataset.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    loader = CodeSearchNetLoader(config)
    train_raw, val_raw, test_raw = loader.load_and_sample()
    
    train_data = loader.prepare_dataset(train_raw)
    val_data = loader.prepare_dataset(val_raw)
    test_data = loader.prepare_dataset(test_raw)
    
    return train_data, val_data, test_data
