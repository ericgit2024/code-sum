# Source Code Summarization with Gemma 2B + LoRA + RAG + Reflective Agent

A novel approach to code summarization combining:
- **Gemma 2B** with 4-bit quantization and LoRA fine-tuning
- **Multi-view structural analysis**: AST, CFG, and PDG extraction
- **RAG retrieval** for context-aware generation
- **Reflective Agent** for iterative summary refinement

## Project Structure

```
.
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── .env.example               # Environment variables template
├── train.py                   # Main training script
├── evaluate.py                # Evaluation script
├── run_inference.py           # Interactive inference script
└── src/
    ├── data/
    │   ├── dataset_loader.py  # CodeSearchNet dataset loader
    │   └── preprocessor.py    # Data preprocessing pipeline
    ├── structure/
    │   ├── ast_extractor.py   # AST extraction and encoding
    │   ├── cfg_extractor.py   # CFG extraction and encoding
    │   └── pdg_extractor.py   # PDG extraction and encoding
    ├── rag/
    │   └── rag_system.py      # RAG retrieval system
    ├── model/
    │   ├── model_loader.py    # Gemma model loader with LoRA
    │   ├── trainer.py         # Training logic
    │   └── inference.py       # Inference pipeline
    ├── agent/
    │   └── reflective_agent.py # Reflective agent for refinement
    └── evaluation/
        └── metrics.py         # BLEU, ROUGE, METEOR metrics
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

Copy `.env.example` to `.env` and add your HuggingFace token:

```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

Or pass the token directly when running scripts:

```bash
python train.py --hf_token YOUR_TOKEN_HERE
```

## Usage

### Training

Train the model on 3000 samples from CodeSearchNet Python dataset with 2 epochs:

```bash
python train.py --hf_token YOUR_HF_TOKEN
```

This will:
1. Load 3000 samples from CodeSearchNet Python
2. Build RAG index from training data
3. Extract AST, CFG, and PDG for each sample
4. Fine-tune Gemma 2B with LoRA (4-bit quantization)
5. Save the trained model to `./outputs/final_model`

### Evaluation

Evaluate the trained model on test set:

```bash
python evaluate.py --checkpoint ./outputs/final_model --hf_token YOUR_HF_TOKEN
```

This will:
1. Load the test dataset
2. Generate summaries using RAG + Reflective Agent
3. Calculate BLEU, ROUGE, and METEOR scores
4. Save results to `evaluation_results/results.json`

To disable the reflective agent:

```bash
python evaluate.py --checkpoint ./outputs/final_model --no_reflective_agent
```

### Inference

Generate summary for a single code snippet:

```bash
python run_inference.py --checkpoint ./outputs/final_model --code "def add(a, b): return a + b"
```

Or from a file:

```bash
python run_inference.py --checkpoint ./outputs/final_model --code_file path/to/code.py
```

## Configuration

Edit `config.yaml` to customize:

- **Dataset**: Sample size, splits
- **Model**: Quantization settings, LoRA parameters
- **Training**: Batch size, learning rate, epochs
- **RAG**: Embedding model, top-k retrieval
- **Reflective Agent**: Max iterations, evaluation criteria
- **Prompts**: System prompt, instruction templates

## Key Features

### 1. Multi-View Structural Analysis

The system extracts three complementary code representations:
- **AST**: Syntax structure
- **CFG**: Control flow
- **PDG**: Data dependencies

These are linearized and fed to the LLM for comprehensive understanding.

### 2. RAG Retrieval

Uses FAISS-based semantic search to retrieve similar code examples from the training set, providing context for better summarization.

### 3. Reflective Agent

Iteratively critiques and refines summaries based on:
- Completeness
- Clarity
- Accuracy
- Conciseness

### 4. Efficient Training

- 4-bit quantization reduces memory usage
- LoRA enables efficient fine-tuning
- Only ~2% of parameters are trainable

## Evaluation Metrics

The system reports:
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: N-gram overlap
- **ROUGE-1, ROUGE-2, ROUGE-L**: Recall-oriented metrics
- **METEOR**: Semantic similarity with synonyms

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- HuggingFace account with Gemma access

## Notes

- First run will download the Gemma 2B model (~5GB)
- CodeSearchNet dataset will be cached locally
- Training takes approximately 2-4 hours on a single GPU
- All outputs are saved to `./outputs` directory

## Citation

If you use this code for your research, please cite:

```
@mastersthesis{your_thesis,
  title={Source Code Summarization using Gemma 2B with LoRA, RAG, and Reflective Agents},
  author={Your Name},
  year={2024},
  school={Your University}
}
```
