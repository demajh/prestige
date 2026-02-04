"""Streaming dataset support for large-scale deduplication.

This module provides streaming deduplication for datasets too large to fit
in memory, using PyTorch IterableDataset and HuggingFace streaming datasets.
"""

from typing import Any, Callable, Dict, Iterator, Optional, Union

from .config import DedupConfig, DedupMode
from .dedup_store import DedupStore
from .metrics import DedupMetrics
from .utils import extract_text, make_key

# Try to import PyTorch
try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader, get_worker_info

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    IterableDataset = object
    DataLoader = None

# Try to import HuggingFace datasets
try:
    from datasets import IterableDataset as HFIterableDataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HFIterableDataset = None


class StreamingDedupDataset(IterableDataset):
    """PyTorch IterableDataset with streaming deduplication.

    Suitable for large datasets that don't fit in memory. Deduplication
    happens on-the-fly during iteration.

    Supports multi-worker DataLoaders with proper worker sharding.

    Example:
        >>> def data_generator():
        ...     for line in open("large_file.txt"):
        ...         yield {"text": line.strip()}
        >>>
        >>> dataset = StreamingDedupDataset(data_generator, config)
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>> for batch in loader:
        ...     # Process deduplicated batch
        ...     pass
    """

    def __init__(
        self,
        data_iterator_fn: Callable[[], Iterator[Any]],
        config: Optional[DedupConfig] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        text_column: Optional[str] = None,
    ):
        """Initialize streaming dedup dataset.

        Args:
            data_iterator_fn: Factory function that returns a fresh iterator
            config: DedupConfig for settings
            transform: Optional transform to apply to items
            text_column: Column containing text (overrides config)

        Raises:
            ImportError: If PyTorch is not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for StreamingDedupDataset. "
                "Install with: pip install torch"
            )

        self.data_iterator_fn = data_iterator_fn
        self.config = config or DedupConfig()
        self.transform = transform

        if text_column:
            self.config.text_column = text_column

        self._metrics = DedupMetrics()

    def __iter__(self) -> Iterator[Any]:
        """Iterate over deduplicated items.

        Yields:
            Non-duplicate items from the data iterator
        """
        # Handle multi-worker case
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Create worker-specific config
        worker_config = DedupConfig(
            mode=self.config.mode,
            semantic_threshold=self.config.semantic_threshold,
            semantic_model_type=self.config.semantic_model_type,
            semantic_model_path=self.config.semantic_model_path,
            semantic_device=self.config.semantic_device,
            text_column=self.config.text_column,
            batch_size=self.config.batch_size,
            key_prefix=f"w{worker_id}_",
        )

        self._metrics = DedupMetrics()
        self._metrics.start()

        with DedupStore(worker_config) as store:
            idx = 0
            for item in self.data_iterator_fn():
                # Shard across workers
                if num_workers > 1 and idx % num_workers != worker_id:
                    idx += 1
                    continue

                text = extract_text(item, self.config.text_column)
                key = f"stream_{worker_id}_{idx}"

                is_duplicate, _ = store.check_and_add(key, text)
                self._metrics.record_item(is_duplicate, index=idx, key=key)
                idx += 1

                if not is_duplicate:
                    if self.transform:
                        item = self.transform(item)
                    yield item

                if idx % self.config.batch_size == 0:
                    store.flush()

            # Update final metrics
            self._metrics.update_from_store_health(store.get_health())

        self._metrics.finish()

    def get_metrics(self) -> DedupMetrics:
        """Get metrics (only accurate after full iteration)."""
        return self._metrics


class DynamicDedupIterator:
    """Iterator wrapper that filters duplicates on-the-fly.

    A simpler alternative to StreamingDedupDataset when you just need
    to wrap an existing iterator.

    Example:
        >>> iterator = DynamicDedupIterator(data_iter, config)
        >>> for item in iterator:
        ...     # Process deduplicated item
        ...     pass
        >>> print(iterator.get_metrics().summary())
    """

    def __init__(
        self,
        data_iterator: Iterator[Any],
        config: Optional[DedupConfig] = None,
        text_column: Optional[str] = None,
    ):
        """Initialize the iterator.

        Args:
            data_iterator: Source iterator
            config: DedupConfig for settings
            text_column: Column containing text
        """
        self.data_iterator = data_iterator
        self.config = config or DedupConfig()

        if text_column:
            self.config.text_column = text_column

        self._store: Optional[DedupStore] = None
        self._metrics = DedupMetrics()
        self._idx = 0

    def __iter__(self) -> Iterator[Any]:
        """Iterate over deduplicated items."""
        self._metrics.start()
        self._store = DedupStore(self.config)
        self._store.open()
        self._idx = 0

        try:
            for item in self.data_iterator:
                text = extract_text(item, self.config.text_column)
                key = f"dynamic_{self._idx}"

                is_duplicate, _ = self._store.check_and_add(key, text)
                self._metrics.record_item(is_duplicate, index=self._idx, key=key)
                self._idx += 1

                if not is_duplicate:
                    yield item

                if self._idx % self.config.batch_size == 0:
                    self._store.flush()

        finally:
            self._metrics.finish()
            if self._store:
                self._metrics.update_from_store_health(self._store.get_health())
                self._store.close()
                self._store = None

    def get_metrics(self) -> DedupMetrics:
        """Get metrics."""
        return self._metrics


def create_streaming_dataloader(
    data_source: Union[Callable[[], Iterator[Any]], "HFIterableDataset"],
    config: Optional[DedupConfig] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    text_column: str = "text",
    transform: Optional[Callable] = None,
    **dataloader_kwargs,
) -> "DataLoader":
    """Create a DataLoader from a streaming source with deduplication.

    Works with both iterator factories and HuggingFace IterableDatasets.

    Args:
        data_source: Factory function returning iterator, or HF IterableDataset
        config: DedupConfig for settings
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes
        text_column: Column containing text
        transform: Optional transform for items
        **dataloader_kwargs: Additional DataLoader arguments

    Returns:
        PyTorch DataLoader

    Example:
        >>> from datasets import load_dataset
        >>> ds = load_dataset("c4", "en", streaming=True)
        >>> loader = create_streaming_dataloader(ds["train"], batch_size=64)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for create_streaming_dataloader")

    if config is None:
        config = DedupConfig()
    config.text_column = text_column

    # Handle HuggingFace IterableDataset
    if HF_AVAILABLE and isinstance(data_source, HFIterableDataset):

        def make_iterator():
            return iter(data_source)

        data_iterator_fn = make_iterator
    elif callable(data_source):
        data_iterator_fn = data_source
    else:
        raise TypeError(
            "data_source must be a callable returning an iterator or HF IterableDataset"
        )

    dataset = StreamingDedupDataset(
        data_iterator_fn, config, transform=transform, text_column=text_column
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **dataloader_kwargs,
    )


def deduplicate_iterator(
    iterator: Iterator[Any],
    mode: str = "exact",
    threshold: float = 0.85,
    text_column: str = "text",
    **kwargs,
) -> Iterator[Any]:
    """Wrap an iterator with deduplication.

    Simple function to add deduplication to any iterator.

    Args:
        iterator: Source iterator
        mode: "exact" or "semantic"
        threshold: Semantic similarity threshold
        text_column: Column containing text
        **kwargs: Additional DedupConfig options

    Yields:
        Deduplicated items

    Example:
        >>> lines = ({"text": line} for line in open("data.txt"))
        >>> for item in deduplicate_iterator(lines, mode="semantic"):
        ...     print(item["text"])
    """
    config = DedupConfig(
        mode=DedupMode.SEMANTIC if mode == "semantic" else DedupMode.EXACT,
        semantic_threshold=threshold,
        text_column=text_column,
        **kwargs,
    )

    wrapper = DynamicDedupIterator(iterator, config)
    yield from wrapper


class ChunkedDedupProcessor:
    """Process large datasets in chunks with deduplication.

    Useful when you can't stream but need to process data in manageable
    chunks to avoid memory issues.

    Example:
        >>> processor = ChunkedDedupProcessor(config, chunk_size=10000)
        >>> for chunk in processor.process_chunks(data_iterator):
        ...     # Process deduplicated chunk
        ...     save_to_disk(chunk)
    """

    def __init__(
        self,
        config: Optional[DedupConfig] = None,
        chunk_size: int = 10000,
    ):
        """Initialize chunked processor.

        Args:
            config: DedupConfig for settings
            chunk_size: Number of items per chunk
        """
        self.config = config or DedupConfig()
        self.chunk_size = chunk_size
        self._store: Optional[DedupStore] = None
        self._metrics = DedupMetrics()

    def process_chunks(
        self,
        data_iterator: Iterator[Any],
        text_column: Optional[str] = None,
    ) -> Iterator[list]:
        """Process data in chunks, yielding deduplicated chunks.

        Args:
            data_iterator: Source iterator
            text_column: Column containing text

        Yields:
            Lists of deduplicated items (chunk_size or smaller)
        """
        text_col = text_column or self.config.text_column

        self._metrics.start()
        self._store = DedupStore(self.config)
        self._store.open()

        try:
            chunk = []
            idx = 0

            for item in data_iterator:
                text = extract_text(item, text_col)
                key = f"chunk_{idx}"

                is_duplicate, _ = self._store.check_and_add(key, text)
                self._metrics.record_item(is_duplicate, index=idx, key=key)
                idx += 1

                if not is_duplicate:
                    chunk.append(item)

                if len(chunk) >= self.chunk_size:
                    yield chunk
                    chunk = []

                if idx % self.config.batch_size == 0:
                    self._store.flush()

            # Yield remaining items
            if chunk:
                yield chunk

        finally:
            self._metrics.finish()
            if self._store:
                self._metrics.update_from_store_health(self._store.get_health())
                self._store.close()
                self._store = None

    def get_metrics(self) -> DedupMetrics:
        """Get processing metrics."""
        return self._metrics
