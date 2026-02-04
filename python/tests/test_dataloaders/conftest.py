"""Pytest fixtures for dataloaders tests."""

import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

import prestige
from prestige.dataloaders import DedupConfig, DedupMode, DedupStore

# Check for optional dependencies
try:
    import torch
    from torch.utils.data import DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from datasets import Dataset as HFDataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


# Skip markers
skip_if_no_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch not available"
)

skip_if_no_hf = pytest.mark.skipif(
    not HF_AVAILABLE, reason="HuggingFace datasets not available"
)

skip_if_no_semantic = pytest.mark.skipif(
    not prestige.SEMANTIC_AVAILABLE, reason="Semantic deduplication not available"
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test databases."""
    path = Path(tempfile.mkdtemp(prefix="prestige_dataloader_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def store_path(temp_dir):
    """Provide a store path within temp directory."""
    return temp_dir / "test_store"


@pytest.fixture
def exact_config(store_path):
    """Create a DedupConfig for exact deduplication."""
    return DedupConfig(
        mode=DedupMode.EXACT,
        store_path=store_path,
        text_column="text",
        batch_size=10,
    )


@pytest.fixture
def semantic_config(store_path):
    """Create a DedupConfig for semantic deduplication."""
    # Use the bge-small model from the cache
    model_path = Path.home() / ".cache" / "prestige" / "models" / "bge-small" / "model.onnx"

    return DedupConfig(
        mode=DedupMode.SEMANTIC,
        semantic_threshold=0.85,
        semantic_model_path=model_path if model_path.exists() else None,
        store_path=store_path,
        text_column="text",
        batch_size=10,
    )


@pytest.fixture
def sample_data() -> List[Dict[str, Any]]:
    """Sample data with some duplicates."""
    return [
        {"text": "The quick brown fox jumps over the lazy dog", "label": 0},
        {"text": "Hello world, this is a test", "label": 1},
        {"text": "The quick brown fox jumps over the lazy dog", "label": 0},  # Exact dup
        {"text": "Python is a great programming language", "label": 2},
        {"text": "Hello world, this is a test", "label": 1},  # Exact dup
        {"text": "Machine learning is fascinating", "label": 3},
        {"text": "Deep learning and neural networks", "label": 4},
        {"text": "The quick brown fox jumps over the lazy dog", "label": 0},  # Exact dup
    ]


@pytest.fixture
def unique_data() -> List[Dict[str, Any]]:
    """Sample data with all unique items."""
    return [
        {"text": "First unique sentence about cats", "label": 0},
        {"text": "Second unique sentence about dogs", "label": 1},
        {"text": "Third unique sentence about birds", "label": 2},
        {"text": "Fourth unique sentence about fish", "label": 3},
        {"text": "Fifth unique sentence about horses", "label": 4},
    ]


@pytest.fixture
def train_data() -> List[Dict[str, Any]]:
    """Sample training data for contamination tests."""
    return [
        {"text": "Training example one about machine learning", "label": 0},
        {"text": "Training example two about deep learning", "label": 1},
        {"text": "Training example three about neural networks", "label": 2},
        {"text": "Test contamination: this should be in test set", "label": 3},
        {"text": "Another training example about AI", "label": 4},
    ]


@pytest.fixture
def test_data() -> List[Dict[str, Any]]:
    """Sample test data for contamination tests."""
    return [
        {"text": "Test example one about data science", "label": 0},
        {"text": "Test contamination: this should be in test set", "label": 1},  # Same as train
        {"text": "Test example three about statistics", "label": 2},
    ]


@pytest.fixture
def dedup_store(exact_config):
    """Create a DedupStore with exact deduplication."""
    store = DedupStore(exact_config)
    store.open()
    yield store
    store.close()


@pytest.fixture
def hf_dataset(sample_data):
    """Create a HuggingFace dataset from sample data."""
    if not HF_AVAILABLE:
        pytest.skip("HuggingFace datasets not available")
    return HFDataset.from_list(sample_data)


@pytest.fixture
def hf_unique_dataset(unique_data):
    """Create a HuggingFace dataset with unique data."""
    if not HF_AVAILABLE:
        pytest.skip("HuggingFace datasets not available")
    return HFDataset.from_list(unique_data)


@pytest.fixture
def hf_train_dataset(train_data):
    """Create a HuggingFace training dataset."""
    if not HF_AVAILABLE:
        pytest.skip("HuggingFace datasets not available")
    return HFDataset.from_list(train_data)


@pytest.fixture
def hf_test_dataset(test_data):
    """Create a HuggingFace test dataset."""
    if not HF_AVAILABLE:
        pytest.skip("HuggingFace datasets not available")
    return HFDataset.from_list(test_data)
