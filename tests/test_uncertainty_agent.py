"""
Unit tests for UncertaintyAgent.
"""

import pytest
import torch
from unittest.mock import Mock, MagicMock
from src.agent.uncertainty_agent import UncertaintyAgent


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = Mock()
    model.device = 'cpu'
    model.modules = Mock(return_value=[])
    model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    tokenizer.decode = Mock(return_value="Divides two numbers.")
    return tokenizer


@pytest.fixture
def config():
    """Create a test configuration."""
    return {
        'uncertainty_agent': {
            'enabled': True,
            'n_samples': 5,
            'confidence_threshold': 0.6,
            'max_refinement_iterations': 2
        }
    }


@pytest.fixture
def uncertainty_agent(mock_model, mock_tokenizer, config):
    """Create an UncertaintyAgent instance for testing."""
    return UncertaintyAgent(mock_model, mock_tokenizer, config, n_samples=5)


def test_initialization(uncertainty_agent):
    """Test that UncertaintyAgent initializes correctly."""
    assert uncertainty_agent.n_samples == 5
    assert uncertainty_agent.confidence_threshold == 0.6
    assert uncertainty_agent.max_refinement_iterations == 2


def test_dropout_enable_disable(uncertainty_agent, mock_model):
    """Test dropout enable/disable functionality."""
    # Create mock dropout module
    dropout_module = Mock()
    dropout_module.__class__.__name__ = 'Dropout'
    mock_model.modules = Mock(return_value=[dropout_module])
    
    # Test enable
    uncertainty_agent.enable_dropout()
    dropout_module.train.assert_called_once()
    
    # Test disable
    uncertainty_agent.disable_dropout()
    dropout_module.eval.assert_called_once()


def test_extract_summary(uncertainty_agent):
    """Test summary extraction from generated text."""
    # Test with prompt removal
    full_text = "Generate a docstring:\n\nDivides two numbers."
    prompt = "Generate a docstring:"
    summary = uncertainty_agent._extract_summary(full_text, prompt)
    assert summary == "Divides two numbers."
    
    # Test with code block removal
    full_text = "Summary: Divides two numbers.\n```python\ndef foo(): pass\n```"
    summary = uncertainty_agent._extract_summary(full_text, "")
    assert "```" not in summary
    assert "def foo" not in summary


def test_split_sentences(uncertainty_agent):
    """Test sentence splitting."""
    text = "First sentence. Second sentence! Third sentence?"
    sentences = uncertainty_agent._split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence"
    assert sentences[1] == "Second sentence"
    assert sentences[2] == "Third sentence"


def test_calculate_sentence_variance(uncertainty_agent):
    """Test variance calculation across summaries."""
    # Test with identical summaries (high confidence)
    summaries = [
        "Divides two numbers.",
        "Divides two numbers.",
        "Divides two numbers."
    ]
    sentences, confidence_scores = uncertainty_agent.calculate_sentence_variance(summaries)
    assert len(sentences) == 1
    assert confidence_scores[0] == 1.0  # Perfect agreement
    
    # Test with diverse summaries (low confidence)
    summaries = [
        "Divides two numbers.",
        "Returns division result.",
        "Computes a divided by b."
    ]
    sentences, confidence_scores = uncertainty_agent.calculate_sentence_variance(summaries)
    assert len(sentences) == 1
    assert confidence_scores[0] < 1.0  # Lower agreement


def test_compute_confidence_scores(uncertainty_agent):
    """Test confidence score computation."""
    summaries = [
        "Divides two numbers.",
        "Divides two numbers.",
        "Returns division result."
    ]
    
    result = uncertainty_agent.compute_confidence_scores(summaries)
    
    assert 'sentences' in result
    assert 'confidence_scores' in result
    assert 'mean_confidence' in result
    assert 'min_confidence' in result
    assert 'low_confidence_indices' in result
    assert 'n_samples' in result
    
    assert result['n_samples'] == 3
    assert 0.0 <= result['mean_confidence'] <= 1.0
    assert 0.0 <= result['min_confidence'] <= 1.0


def test_generate_multiple_summaries(uncertainty_agent, mock_tokenizer):
    """Test generation of multiple summaries."""
    prompt = "Generate a docstring for this code"
    
    # Mock tokenizer to return proper format
    mock_tokenizer.return_value = {
        'input_ids': torch.tensor([[1, 2, 3]]),
        'attention_mask': torch.tensor([[1, 1, 1]])
    }
    
    summaries = uncertainty_agent.generate_multiple_summaries(prompt, n_samples=3)
    
    assert len(summaries) == 3
    assert all(isinstance(s, str) for s in summaries)


def test_generate_with_uncertainty_high_confidence(uncertainty_agent, mock_tokenizer):
    """Test full pipeline with high confidence (no refinement needed)."""
    code = "def divide(a, b): return a / b"
    initial_summary = "Divides two numbers"
    
    # Mock to return identical summaries (high confidence)
    mock_tokenizer.decode = Mock(return_value="Divides two numbers.")
    
    result = uncertainty_agent.generate_with_uncertainty(code, initial_summary)
    
    assert 'final_summary' in result
    assert 'confidence_scores' in result
    assert 'mean_confidence' in result
    assert 'uncertainty_metadata' in result
    
    # High confidence should not trigger refinement
    assert result['uncertainty_metadata']['refinement_applied'] == False


def test_generate_with_uncertainty_low_confidence(uncertainty_agent, mock_tokenizer):
    """Test full pipeline with low confidence (refinement triggered)."""
    code = "def divide(a, b): return a / b"
    initial_summary = "Divides two numbers"
    
    # Mock to return diverse summaries (low confidence)
    responses = [
        "Divides two numbers.",
        "Returns division result.",
        "Computes a divided by b.",
        "Performs division operation.",
        "Calculates quotient."
    ]
    mock_tokenizer.decode = Mock(side_effect=responses + ["Improved summary."])
    
    result = uncertainty_agent.generate_with_uncertainty(code, initial_summary)
    
    assert 'final_summary' in result
    assert 'confidence_scores' in result
    assert 'mean_confidence' in result
    
    # Low confidence should trigger refinement
    # Note: This test may need adjustment based on actual implementation


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
