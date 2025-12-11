"""
Setup script for code summarization project.
"""

from setuptools import setup, find_packages

setup(
    name="code-summarization",
    version="0.1.0",
    description="Source Code Summarization with Gemma 2B + LoRA + RAG + Reflective Agent",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "bitsandbytes>=0.41.0",
        "accelerate>=0.25.0",
        "datasets>=2.14.0",
        "huggingface-hub>=0.19.0",
        "networkx>=3.1",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "nltk>=3.8.1",
        "rouge-score>=0.1.2",
        "evaluate>=0.4.1",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
    ],
)
