"""Utility functions for deduplicated dataloaders.

This module provides helper functions for text extraction, key generation,
and other common operations used across the dataloaders package.
"""

from typing import Any, Callable, Dict, List, Optional, Union
import hashlib


def extract_text(
    item: Any,
    text_column: str = "text",
    text_columns: Optional[List[str]] = None,
    separator: str = " ",
) -> str:
    """Extract text from an item for deduplication.

    Supports dictionaries, objects with attributes, and direct string values.
    Can extract from multiple columns and concatenate them.

    Args:
        item: The item to extract text from
        text_column: Primary column/attribute name containing text
        text_columns: Optional list of columns to concatenate
        separator: Separator when concatenating multiple columns

    Returns:
        Extracted text as a string

    Example:
        >>> extract_text({"text": "hello world"})
        'hello world'
        >>> extract_text({"q1": "foo", "q2": "bar"}, text_columns=["q1", "q2"])
        'foo bar'
    """
    if text_columns:
        parts = []
        for col in text_columns:
            parts.append(_get_value(item, col))
        return separator.join(parts)

    return _get_value(item, text_column)


def _get_value(item: Any, key: str) -> str:
    """Get a value from an item by key/attribute name.

    Args:
        item: The item to get value from
        key: The key or attribute name

    Returns:
        Value as string, or empty string if not found
    """
    if isinstance(item, dict):
        value = item.get(key, "")
    elif hasattr(item, key):
        value = getattr(item, key)
    elif isinstance(item, str):
        return item
    else:
        value = str(item)

    return str(value) if value is not None else ""


def make_key(
    index: int,
    prefix: str = "",
    namespace: Optional[str] = None,
) -> str:
    """Generate a unique key for an item.

    Args:
        index: Item index
        prefix: Key prefix (e.g., "train_", "test_")
        namespace: Optional additional namespace

    Returns:
        Generated key string

    Example:
        >>> make_key(42, prefix="train_")
        'train_item_42'
        >>> make_key(0, namespace="dataset1")
        'dataset1_item_0'
    """
    parts = []
    if namespace:
        parts.append(namespace)
    if prefix:
        parts.append(prefix)
    parts.append(f"item_{index}")
    return "_".join(parts) if len(parts) > 1 else parts[0]


def hash_text(text: str) -> str:
    """Generate a short hash of text for cache keys.

    Args:
        text: Text to hash

    Returns:
        16-character hex hash

    Example:
        >>> hash_text("hello world")
        'b94d27b9934d3e08'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_config(config: Any) -> str:
    """Generate a hash of configuration for cache keys.

    Args:
        config: Configuration object (DedupConfig or similar)

    Returns:
        8-character hex hash representing the configuration

    Example:
        >>> config = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
        >>> hash_config(config)
        'a1b2c3d4'
    """
    # Extract key configuration values that affect deduplication results
    if hasattr(config, "mode"):
        mode_str = str(config.mode.value if hasattr(config.mode, "value") else config.mode)
    else:
        mode_str = "exact"

    threshold = getattr(config, "semantic_threshold", 0.85)
    model_type = getattr(config, "semantic_model_type", "")
    reranker = getattr(config, "enable_reranker", False)
    rnn = getattr(config, "enable_rnn", False)
    margin = getattr(config, "enable_margin_gating", False)

    config_str = f"{mode_str}_{threshold}_{model_type}_{reranker}_{rnn}_{margin}"
    return hashlib.sha256(config_str.encode()).hexdigest()[:8]


def batched(iterable, batch_size: int):
    """Yield batches from an iterable.

    Args:
        iterable: Any iterable
        batch_size: Size of each batch

    Yields:
        Lists of items of at most batch_size length

    Example:
        >>> list(batched([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_text_extractor(
    text_column: str = "text",
    text_columns: Optional[List[str]] = None,
    separator: str = " ",
    preprocess: Optional[Callable[[str], str]] = None,
) -> Callable[[Any], str]:
    """Create a text extraction function with fixed configuration.

    Useful for creating an extractor once and reusing it.

    Args:
        text_column: Primary column name
        text_columns: Optional list of columns to concatenate
        separator: Separator for multiple columns
        preprocess: Optional preprocessing function to apply to extracted text

    Returns:
        A function that extracts text from items

    Example:
        >>> extractor = create_text_extractor(text_column="content")
        >>> extractor({"content": "hello"})
        'hello'
    """

    def extractor(item: Any) -> str:
        text = extract_text(item, text_column, text_columns, separator)
        if preprocess:
            text = preprocess(text)
        return text

    return extractor


def estimate_memory_usage(
    num_items: int,
    avg_text_length: int = 500,
    mode: str = "exact",
    embedding_dim: int = 384,
) -> Dict[str, int]:
    """Estimate memory usage for deduplication.

    Provides rough estimates for planning purposes.

    Args:
        num_items: Number of items to process
        avg_text_length: Average text length in characters
        mode: "exact" or "semantic"
        embedding_dim: Embedding dimension for semantic mode

    Returns:
        Dictionary with estimated memory usage in bytes

    Example:
        >>> estimate_memory_usage(1_000_000, mode="semantic")
        {'text_bytes': 500000000, 'embedding_bytes': 1536000000, ...}
    """
    # Text storage (rough estimate)
    text_bytes = num_items * avg_text_length

    # Key storage (avg 30 bytes per key)
    key_bytes = num_items * 30

    # Object ID storage (16 bytes per ID)
    object_id_bytes = num_items * 16

    # For exact mode: SHA-256 hash index
    hash_bytes = num_items * 32 if mode == "exact" else 0

    # For semantic mode: embeddings (4 bytes per float)
    embedding_bytes = num_items * embedding_dim * 4 if mode == "semantic" else 0

    # HNSW index overhead (roughly 100 bytes per item for M=16)
    hnsw_bytes = num_items * 100 if mode == "semantic" else 0

    total = text_bytes + key_bytes + object_id_bytes + hash_bytes + embedding_bytes + hnsw_bytes

    return {
        "text_bytes": text_bytes,
        "key_bytes": key_bytes,
        "object_id_bytes": object_id_bytes,
        "hash_bytes": hash_bytes,
        "embedding_bytes": embedding_bytes,
        "hnsw_bytes": hnsw_bytes,
        "total_bytes": total,
        "total_mb": total / (1024 * 1024),
        "total_gb": total / (1024 * 1024 * 1024),
    }


def validate_text_column(
    sample_item: Any,
    text_column: str,
    text_columns: Optional[List[str]] = None,
) -> bool:
    """Validate that text column(s) exist in a sample item.

    Args:
        sample_item: A sample item from the dataset
        text_column: Primary column name to check
        text_columns: Optional list of columns to check

    Returns:
        True if all columns exist and contain string-convertible values

    Raises:
        ValueError: If columns are missing or invalid
    """
    columns_to_check = text_columns if text_columns else [text_column]

    for col in columns_to_check:
        if isinstance(sample_item, dict):
            if col not in sample_item:
                raise ValueError(
                    f"Column '{col}' not found in item. "
                    f"Available columns: {list(sample_item.keys())}"
                )
        elif not hasattr(sample_item, col):
            raise ValueError(f"Attribute '{col}' not found in item of type {type(sample_item)}")

    return True
