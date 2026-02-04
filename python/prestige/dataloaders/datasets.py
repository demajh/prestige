"""PyTorch Dataset wrappers with deduplication support.

This module provides PyTorch Dataset classes that automatically deduplicate
data during access, enabling clean training without duplicate examples.
"""

from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Union

from .config import DedupConfig, DedupMode
from .dedup_store import DedupStore
from .metrics import DedupMetrics
from .utils import extract_text, make_key

# Try to import torch - make it optional
try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Fallback for type hints
    DataLoader = None


class DedupDataset(Dataset):
    """PyTorch Dataset wrapper that provides deduplicated access to data.

    This class wraps any sequence-like data and filters out duplicates during
    initialization. The deduplication is computed upfront, creating an index
    of valid (non-duplicate) items for efficient random access.

    Supports both exact (SHA-256) and semantic (embedding-based) deduplication.

    Attributes:
        config: DedupConfig controlling deduplication behavior
        transform: Optional transform to apply to items

    Example:
        >>> from torch.utils.data import DataLoader
        >>> config = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
        >>> dataset = DedupDataset(train_data, config, text_column="content")
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in loader:
        ...     # Train on deduplicated data
        ...     pass
    """

    def __init__(
        self,
        data: Sequence[Any],
        config: Optional[DedupConfig] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        text_column: Optional[str] = None,
        precompute: bool = True,
        verbose: bool = False,
    ):
        """Initialize the deduplicated dataset.

        Args:
            data: Sequence of items (list, HuggingFace dataset, etc.)
            config: DedupConfig for deduplication settings (default: exact mode)
            transform: Optional transform to apply to items after retrieval
            text_column: Column containing text (overrides config.text_column)
            precompute: If True, compute dedup index immediately
            verbose: If True, print progress information

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DedupDataset. "
                "Install with: pip install torch"
            )

        self._data = data
        self.config = config or DedupConfig()
        self.transform = transform
        self._verbose = verbose

        # Override text column if provided
        if text_column:
            self.config.text_column = text_column

        # Index of valid (non-duplicate) items
        self._valid_indices: Optional[List[int]] = None

        # Metrics
        self._metrics = DedupMetrics()
        self._store_metrics: Optional[Dict[str, Any]] = None

        if precompute:
            self._precompute_valid_indices()

    def _precompute_valid_indices(self) -> None:
        """Build index of non-duplicate items."""
        self._metrics.start()
        self._valid_indices = []

        with DedupStore(self.config) as store:
            total = len(self._data)

            for idx in range(total):
                item = self._data[idx]
                text = extract_text(item, self.config.text_column)
                key = make_key(idx, prefix=self.config.key_prefix)

                is_duplicate, _ = store.check_and_add(key, text)
                self._metrics.record_item(is_duplicate, index=idx, key=key)

                if not is_duplicate:
                    self._valid_indices.append(idx)

                # Periodic flush
                if (idx + 1) % self.config.batch_size == 0:
                    store.flush()
                    if self._verbose:
                        kept = len(self._valid_indices)
                        removed = idx + 1 - kept
                        print(
                            f"Processed {idx + 1}/{total}: "
                            f"kept {kept}, removed {removed} duplicates"
                        )

            # Capture final store metrics
            self._store_metrics = store.get_health()
            self._metrics.update_from_store_health(self._store_metrics)

        self._metrics.finish()

        if self._verbose:
            print(self._metrics.summary())

    def _ensure_computed(self) -> None:
        """Ensure valid indices are computed."""
        if self._valid_indices is None:
            self._precompute_valid_indices()

    def __len__(self) -> int:
        """Return number of unique (non-duplicate) items."""
        self._ensure_computed()
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> Any:
        """Get item by deduplicated index.

        Args:
            idx: Index in the deduplicated dataset

        Returns:
            Item at the given index (after any transform)
        """
        self._ensure_computed()

        # Map to original index
        original_idx = self._valid_indices[idx]
        item = self._data[original_idx]

        if self.transform:
            item = self.transform(item)

        return item

    def get_original_index(self, dedup_idx: int) -> int:
        """Get the original index for a deduplicated index.

        Args:
            dedup_idx: Index in the deduplicated dataset

        Returns:
            Corresponding index in the original dataset
        """
        self._ensure_computed()
        return self._valid_indices[dedup_idx]

    def get_metrics(self) -> DedupMetrics:
        """Get deduplication metrics.

        Returns:
            DedupMetrics with deduplication statistics
        """
        self._ensure_computed()
        return self._metrics

    def get_removed_indices(self) -> List[int]:
        """Get indices of removed duplicate items.

        Returns:
            List of original indices that were removed as duplicates
        """
        return self._metrics.removed_indices

    def get_valid_indices(self) -> List[int]:
        """Get indices of kept (valid) items.

        Returns:
            List of original indices that were kept
        """
        self._ensure_computed()
        return list(self._valid_indices)

    @property
    def original_size(self) -> int:
        """Size of the original dataset before deduplication."""
        return len(self._data)

    @property
    def dedup_ratio(self) -> float:
        """Deduplication ratio (original_size / deduplicated_size)."""
        self._ensure_computed()
        if len(self._valid_indices) == 0:
            return 1.0
        return len(self._data) / len(self._valid_indices)


class DedupDatasetView(Dataset):
    """A view of a DedupDataset with additional filtering or transforms.

    This allows creating subsets or applying different transforms without
    recomputing deduplication.

    Example:
        >>> dataset = DedupDataset(data, config)
        >>> train_view = DedupDatasetView(dataset, indices=range(1000))
        >>> val_view = DedupDatasetView(dataset, indices=range(1000, 1200))
    """

    def __init__(
        self,
        parent: DedupDataset,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
    ):
        """Initialize the dataset view.

        Args:
            parent: Parent DedupDataset
            indices: Optional subset of indices to include
            transform: Optional additional transform (applied after parent's)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DedupDatasetView")

        self._parent = parent
        self._indices = list(indices) if indices else list(range(len(parent)))
        self._transform = transform

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Any:
        parent_idx = self._indices[idx]
        item = self._parent[parent_idx]

        if self._transform:
            item = self._transform(item)

        return item


def create_dedup_dataloader(
    data: Sequence[Any],
    config: Optional[DedupConfig] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    text_column: str = "text",
    transform: Optional[Callable] = None,
    **dataloader_kwargs,
) -> "DataLoader":
    """Convenience function to create a DataLoader with deduplication.

    Creates a DedupDataset and wraps it in a PyTorch DataLoader.

    Args:
        data: Sequence of items to deduplicate
        config: DedupConfig for deduplication settings
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        text_column: Column containing text to deduplicate
        transform: Optional transform for items
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        PyTorch DataLoader over deduplicated data

    Example:
        >>> loader = create_dedup_dataloader(
        ...     train_data,
        ...     mode="semantic",
        ...     threshold=0.9,
        ...     batch_size=64,
        ... )
        >>> for batch in loader:
        ...     # Train on batch
        ...     pass
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for create_dedup_dataloader. "
            "Install with: pip install torch"
        )

    if config is None:
        config = DedupConfig()

    config.text_column = text_column

    dataset = DedupDataset(data, config, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **dataloader_kwargs,
    )


def collate_with_dedup_info(
    batch: List[Dict[str, Any]],
    include_dedup_info: bool = True,
) -> Dict[str, Any]:
    """Custom collate function that handles dedup metadata.

    Use this with DataLoader when items include deduplication info.

    Args:
        batch: List of items from the dataset
        include_dedup_info: Whether to include _dedup_* fields

    Returns:
        Collated batch as dictionary
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for collate_with_dedup_info")

    if not batch:
        return {}

    result = {}

    # Get all keys from first item
    if isinstance(batch[0], dict):
        keys = batch[0].keys()

        # Dedup info fields to potentially exclude
        dedup_fields = {"_is_duplicate", "_object_id", "_original_index"}

        for key in keys:
            # Skip dedup fields if not including dedup info
            if not include_dedup_info:
                if key.startswith("_dedup_") or key in dedup_fields:
                    continue

            values = [item[key] for item in batch]

            # Try to stack tensors
            if isinstance(values[0], torch.Tensor):
                try:
                    result[key] = torch.stack(values)
                except Exception:
                    result[key] = values
            else:
                result[key] = values

    return result


class LazyDedupDataset(Dataset):
    """Dataset that performs deduplication lazily during iteration.

    Unlike DedupDataset which precomputes all dedup decisions, this class
    checks deduplication on-the-fly. Useful when you want to:
    - Start training immediately while dedup happens in background
    - Use different dedup behavior per epoch

    Note: This returns all items but marks duplicates in the output.
    Use with a custom sampler or collate function to skip duplicates.

    Example:
        >>> dataset = LazyDedupDataset(data, config)
        >>> for item in dataset:
        ...     if not item.get("_is_duplicate", False):
        ...         # Process non-duplicate
        ...         pass
    """

    def __init__(
        self,
        data: Sequence[Any],
        config: Optional[DedupConfig] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        text_column: Optional[str] = None,
    ):
        """Initialize lazy dedup dataset.

        Args:
            data: Sequence of items
            config: DedupConfig for settings
            transform: Optional transform
            text_column: Column containing text
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LazyDedupDataset")

        self._data = data
        self.config = config or DedupConfig()
        self.transform = transform

        if text_column:
            self.config.text_column = text_column

        self._store: Optional[DedupStore] = None
        self._metrics = DedupMetrics()

    def _ensure_store(self) -> DedupStore:
        """Ensure the dedup store is open."""
        if self._store is None:
            self._store = DedupStore(self.config)
            self._store.open()
            self._metrics.start()
        return self._store

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with dedup info attached.

        Returns:
            Item with added _is_duplicate and _object_id fields
        """
        store = self._ensure_store()

        item = self._data[idx]
        text = extract_text(item, self.config.text_column)
        key = make_key(idx, prefix=self.config.key_prefix)

        is_duplicate, object_id = store.check_and_add(key, text)
        self._metrics.record_item(is_duplicate, index=idx, key=key)

        # Apply transform
        if self.transform:
            item = self.transform(item)

        # Add dedup info
        if isinstance(item, dict):
            result = dict(item)
        else:
            result = {"data": item}

        result["_is_duplicate"] = is_duplicate
        result["_object_id"] = object_id
        result["_original_index"] = idx

        return result

    def get_metrics(self) -> DedupMetrics:
        """Get current metrics."""
        if self._store:
            health = self._store.get_health()
            self._metrics.update_from_store_health(health)
        return self._metrics

    def reset(self) -> None:
        """Reset the store and metrics for a new pass."""
        if self._store:
            self._store.close()
            self._store = None
        self._metrics = DedupMetrics()

    def close(self) -> None:
        """Close the underlying store."""
        if self._store:
            self._metrics.finish()
            self._store.close()
            self._store = None

    def __del__(self):
        self.close()
