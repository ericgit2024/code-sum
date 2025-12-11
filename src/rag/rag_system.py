"""
RAG (Retrieval-Augmented Generation) system for code summarization.
Uses embeddings to retrieve similar code examples.
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple


class RAGSystem:
    """RAG system for retrieving similar code examples."""
    
    def __init__(self, config: Dict):
        """
        Initialize RAG system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.embedding_model_name = config['rag']['embedding_model']
        self.top_k = config['rag']['top_k']
        self.embedding_dim = config['rag']['embedding_dim']
        self.index_path = config['rag']['index_path']
        
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        self.index = None
        self.code_examples = []
        self.summaries = []
        
    def build_index(self, train_data: List[Dict]):
        """
        Build FAISS index from training data.
        
        Args:
            train_data: List of training examples with 'code' and 'docstring'
        """
        print("Building RAG index from training data...")
        
        # Extract code and summaries
        codes = []
        summaries = []
        
        for example in train_data:
            if example['code'] and example['docstring']:
                codes.append(example['code'])
                summaries.append(example['docstring'])
        
        print(f"Encoding {len(codes)} code examples...")
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            codes,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.code_examples = codes
        self.summaries = summaries
        
        print(f"RAG index built with {len(codes)} examples")
        
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.index_path, exist_ok=True)
        
        # Save FAISS index
        index_file = os.path.join(self.index_path, "faiss.index")
        faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                'code_examples': self.code_examples,
                'summaries': self.summaries
            }, f)
        
        print(f"RAG index saved to {self.index_path}")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        index_file = os.path.join(self.index_path, "faiss.index")
        metadata_file = os.path.join(self.index_path, "metadata.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            raise FileNotFoundError(f"RAG index not found at {self.index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(index_file)
        
        # Load metadata
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
            self.code_examples = metadata['code_examples']
            self.summaries = metadata['summaries']
        
        print(f"RAG index loaded from {self.index_path}")
    
    def retrieve(self, query_code: str) -> List[Dict]:
        """
        Retrieve similar code examples.
        
        Args:
            query_code: Code to find similar examples for
            
        Returns:
            List of similar examples with code and summary
        """
        if self.index is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query_code],
            convert_to_numpy=True
        )
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            self.top_k
        )
        
        # Retrieve examples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.code_examples):
                results.append({
                    'code': self.code_examples[idx],
                    'summary': self.summaries[idx],
                    'distance': float(dist)
                })
        
        return results
    
    def format_rag_context(self, retrieved_examples: List[Dict]) -> str:
        """
        Format retrieved examples into context string.
        
        Args:
            retrieved_examples: List of retrieved examples
            
        Returns:
            Formatted context string
        """
        if not retrieved_examples:
            return "No similar examples found."
        
        context_parts = []
        for i, example in enumerate(retrieved_examples, 1):
            context_parts.append(f"Example {i}:")
            context_parts.append(f"Code: {example['code'][:200]}...")
            context_parts.append(f"Summary: {example['summary']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
