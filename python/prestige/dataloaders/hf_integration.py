"""HuggingFace datasets integration for deduplication.

This module provides integration with the HuggingFace datasets library,
enabling easy deduplication of HuggingFace datasets with both static
(pre-filter) and streaming approaches.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import hashlib

from .config import DedupConfig, DedupMode, CacheConfig
from .dedup_store import DedupStore
from .metrics import DedupMetrics
from .utils import extract_text, make_key, hash_config

# Try to import HuggingFace datasets
try:
    from datasets import Dataset as HFDataset
    from datasets import DatasetDict, IterableDataset as HFIterableDataset
    from datasets import load_from_disk

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HFDataset = None
    DatasetDict = None
    HFIterableDataset = None


class HuggingFaceDeduplicator:
    """Integration layer for HuggingFace datasets library.

    Provides methods to deduplicate HuggingFace datasets using prestige's
    deduplication capabilities. Supports both map-style and iterable datasets.

    Example:
        >>> from datasets import load_dataset
        >>> from prestige.dataloaders import HuggingFaceDeduplicator, DedupConfig
        >>>
        >>> config = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
        >>> deduplicator = HuggingFaceDeduplicator(config)
        >>>
        >>> dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        >>> deduped = deduplicator.deduplicate(dataset["train"])
    """

    def __init__(self, config: DedupConfig):
        """Initialize the deduplicator.

        Args:
            config: Configuration for deduplication behavior

        Raises:
            ImportError: If HuggingFace datasets is not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets is required for HuggingFaceDeduplicator. "
                "Install with: pip install datasets"
            )

        self.config = config
        self._metrics = DedupMetrics()

    def deduplicate(
        self,
        dataset: Union["HFDataset", "DatasetDict"],
        text_column: Optional[str] = None,
        desc: str = "Deduplicating",
        verbose: bool = False,
    ) -> Union["HFDataset", "DatasetDict"]:
        """Deduplicate a HuggingFace dataset.

        For DatasetDict, deduplicates each split independently.

        Args:
            dataset: HuggingFace Dataset or DatasetDict
            text_column: Column containing text (overrides config)
            desc: Progress description
            verbose: Print progress information

        Returns:
            Deduplicated dataset (same type as input)

        Example:
            >>> deduped = deduplicator.deduplicate(dataset, text_column="text")
            >>> print(f"Kept {len(deduped)} of {len(dataset)} examples")
        """
        text_col = text_column or self.config.text_column

        if isinstance(dataset, DatasetDict):
            return DatasetDict(
                {
                    split: self._deduplicate_split(
                        ds, text_col, f"{desc} ({split})", verbose
                    )
                    for split, ds in dataset.items()
                }
            )
        else:
            return self._deduplicate_split(dataset, text_col, desc, verbose)

    def _deduplicate_split(
        self,
        dataset: "HFDataset",
        text_column: str,
        desc: str,
        verbose: bool,
    ) -> "HFDataset":
        """Deduplicate a single dataset split.

        Args:
            dataset: HuggingFace Dataset
            text_column: Column containing text
            desc: Progress description
            verbose: Print progress

        Returns:
            Deduplicated dataset
        """
        self._metrics = DedupMetrics()
        self._metrics.start()

        indices_to_keep = []

        with DedupStore(self.config) as store:
            total = len(dataset)

            for idx in range(total):
                example = dataset[idx]
                text = str(example.get(text_column, ""))
                key = f"hf_{idx}"

                is_duplicate, _ = store.check_and_add(key, text)
                self._metrics.record_item(is_duplicate, index=idx)

                if not is_duplicate:
                    indices_to_keep.append(idx)

                if (idx + 1) % self.config.batch_size == 0:
                    store.flush()
                    if verbose:
                        print(
                            f"{desc}: {idx + 1}/{total} processed, "
                            f"{len(indices_to_keep)} unique"
                        )

            # Update metrics from store
            health = store.get_health()
            self._metrics.update_from_store_health(health)

        self._metrics.finish()

        if verbose:
            print(self._metrics.summary())

        # Use select() for efficient filtering
        return dataset.select(indices_to_keep)

    def deduplicate_across_splits(
        self,
        dataset: "DatasetDict",
        text_column: Optional[str] = None,
        primary_split: str = "train",
        verbose: bool = False,
    ) -> "DatasetDict":
        """Deduplicate across all splits, removing items that appear in primary split.

        This is useful for ensuring validation/test sets don't contain
        examples from the training set.

        Args:
            dataset: DatasetDict with multiple splits
            text_column: Column containing text
            primary_split: Split to keep all items from (usually "train")
            verbose: Print progress

        Returns:
            DatasetDict with cross-split deduplication applied

        Example:
            >>> # Remove from val/test any examples that appear in train
            >>> deduped = deduplicator.deduplicate_across_splits(
            ...     dataset, primary_split="train"
            ... )
        """
        text_col = text_column or self.config.text_column
        result = {}

        with DedupStore(self.config) as store:
            # First, index the primary split
            if primary_split in dataset:
                primary = dataset[primary_split]
                if verbose:
                    print(f"Indexing {primary_split} ({len(primary)} examples)...")

                for idx in range(len(primary)):
                    text = str(primary[idx].get(text_col, ""))
                    store.check_and_add(f"{primary_split}_{idx}", text)

                    if (idx + 1) % self.config.batch_size == 0:
                        store.flush()

                store.flush()
                result[primary_split] = primary

            # Then filter other splits
            for split_name, split_data in dataset.items():
                if split_name == primary_split:
                    continue

                if verbose:
                    print(f"Filtering {split_name} ({len(split_data)} examples)...")

                indices_to_keep = []
                for idx in range(len(split_data)):
                    text = str(split_data[idx].get(text_col, ""))

                    # Check if already in store (from primary)
                    is_dup, _ = store.check_contamination(f"check_{idx}", text)

                    if not is_dup:
                        indices_to_keep.append(idx)

                result[split_name] = split_data.select(indices_to_keep)

                if verbose:
                    removed = len(split_data) - len(indices_to_keep)
                    print(f"  Kept {len(indices_to_keep)}, removed {removed}")

        return DatasetDict(result)

    def get_metrics(self) -> DedupMetrics:
        """Get metrics from last deduplication operation.

        Returns:
            DedupMetrics with statistics
        """
        return self._metrics


class StaticDedupPipeline:
    """Pipeline for one-time deduplication with caching.

    Caches deduplicated datasets to disk for fast reloading.

    Example:
        >>> pipeline = StaticDedupPipeline(config, cache_dir="./dedup_cache")
        >>> deduped = pipeline.process_and_cache(dataset, "my_dataset_v1")
        >>> # Next time, loads from cache instantly
        >>> deduped = pipeline.process_and_cache(dataset, "my_dataset_v1")
    """

    def __init__(
        self,
        config: DedupConfig,
        cache_config: Optional[CacheConfig] = None,
    ):
        """Initialize the pipeline.

        Args:
            config: DedupConfig for deduplication
            cache_config: CacheConfig for caching behavior
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets is required for StaticDedupPipeline"
            )

        self.config = config
        self.cache_config = cache_config or CacheConfig()

        # Ensure cache directory exists
        self.cache_config.cache_dir.mkdir(parents=True, exist_ok=True)

    def process_and_cache(
        self,
        dataset: Union["HFDataset", "DatasetDict"],
        dataset_name: str,
        text_column: Optional[str] = None,
        force_reprocess: bool = False,
        verbose: bool = False,
    ) -> Union["HFDataset", "DatasetDict"]:
        """Process dataset and cache the result.

        Args:
            dataset: Input dataset
            dataset_name: Unique name for caching
            text_column: Column containing text
            force_reprocess: Ignore cache and reprocess
            verbose: Print progress

        Returns:
            Deduplicated dataset (from cache if available)
        """
        cache_path = self._get_cache_path(dataset_name)

        # Check cache
        if (
            not force_reprocess
            and not self.cache_config.force_reprocess
            and cache_path.exists()
        ):
            if verbose:
                print(f"Loading from cache: {cache_path}")
            return load_from_disk(str(cache_path / "dataset"))

        # Process
        if verbose:
            print(f"Deduplicating {dataset_name}...")

        deduplicator = HuggingFaceDeduplicator(self.config)
        deduped = deduplicator.deduplicate(
            dataset, text_column=text_column, verbose=verbose
        )
        metrics = deduplicator.get_metrics()

        # Cache
        self._save_to_cache(deduped, metrics, cache_path)

        if verbose:
            print(f"Cached to: {cache_path}")

        return deduped

    def _get_cache_path(self, dataset_name: str) -> Path:
        """Generate cache path based on config and dataset name."""
        config_hash = hash_config(self.config)
        cache_key = self.cache_config.cache_key or f"{dataset_name}_{config_hash}"
        return self.cache_config.cache_dir / cache_key

    def _save_to_cache(
        self,
        dataset: Union["HFDataset", "DatasetDict"],
        metrics: DedupMetrics,
        path: Path,
    ) -> None:
        """Save dataset and metrics to cache."""
        path.mkdir(parents=True, exist_ok=True)

        # Save dataset
        if isinstance(dataset, DatasetDict):
            dataset.save_to_disk(str(path / "dataset"))
        else:
            dataset.save_to_disk(str(path / "dataset"))

        # Save metrics
        if self.cache_config.save_metrics:
            with open(path / "metrics.json", "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)

            # Save config
            config_dict = {
                "mode": self.config.mode.value,
                "threshold": self.config.semantic_threshold,
                "model_type": self.config.semantic_model_type,
            }
            with open(path / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

    def load_cached(self, dataset_name: str) -> Optional["HFDataset"]:
        """Load a cached dataset if available.

        Args:
            dataset_name: Name of the cached dataset

        Returns:
            Cached dataset or None if not found
        """
        cache_path = self._get_cache_path(dataset_name)
        if cache_path.exists():
            return load_from_disk(str(cache_path / "dataset"))
        return None

    def load_metrics(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """Load cached metrics for a dataset.

        Args:
            dataset_name: Name of the cached dataset

        Returns:
            Metrics dictionary or None if not found
        """
        cache_path = self._get_cache_path(dataset_name)
        metrics_path = cache_path / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                return json.load(f)
        return None

    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear cached datasets.

        Args:
            dataset_name: Specific dataset to clear, or None for all
        """
        import shutil

        if dataset_name:
            cache_path = self._get_cache_path(dataset_name)
            if cache_path.exists():
                shutil.rmtree(cache_path)
        else:
            # Clear all
            for item in self.cache_config.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)


# Convenience functions


def deduplicate_dataset(
    dataset: Union["HFDataset", "DatasetDict"],
    mode: str = "exact",
    threshold: float = 0.85,
    text_column: str = "text",
    verbose: bool = False,
    **kwargs,
) -> Union["HFDataset", "DatasetDict"]:
    """One-liner for deduplicating HuggingFace datasets.

    Args:
        dataset: HuggingFace Dataset or DatasetDict
        mode: "exact" or "semantic"
        threshold: Semantic similarity threshold (0.0-1.0)
        text_column: Column containing text to deduplicate
        verbose: Print progress information
        **kwargs: Additional DedupConfig options

    Returns:
        Deduplicated dataset

    Example:
        >>> from datasets import load_dataset
        >>> from prestige.dataloaders import deduplicate_dataset
        >>>
        >>> ds = load_dataset("wikitext", "wikitext-2-raw-v1")
        >>> deduped = deduplicate_dataset(ds["train"], mode="semantic", threshold=0.9)
        >>> print(f"Reduced from {len(ds['train'])} to {len(deduped)} examples")
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace datasets is required. Install with: pip install datasets"
        )

    config = DedupConfig(
        mode=DedupMode.SEMANTIC if mode == "semantic" else DedupMode.EXACT,
        semantic_threshold=threshold,
        text_column=text_column,
        **kwargs,
    )

    deduplicator = HuggingFaceDeduplicator(config)
    return deduplicator.deduplicate(dataset, verbose=verbose)


def deduplicate_and_cache(
    dataset: Union["HFDataset", "DatasetDict"],
    dataset_name: str,
    cache_dir: Optional[Path] = None,
    mode: str = "exact",
    threshold: float = 0.85,
    text_column: str = "text",
    force: bool = False,
    verbose: bool = False,
    **kwargs,
) -> Union["HFDataset", "DatasetDict"]:
    """Deduplicate and cache to disk for fast reloading.

    Args:
        dataset: HuggingFace Dataset or DatasetDict
        dataset_name: Unique name for caching
        cache_dir: Directory for cache (default: ~/.cache/prestige/dedup)
        mode: "exact" or "semantic"
        threshold: Semantic similarity threshold
        text_column: Column containing text
        force: Force reprocessing even if cached
        verbose: Print progress
        **kwargs: Additional DedupConfig options

    Returns:
        Deduplicated dataset

    Example:
        >>> deduped = deduplicate_and_cache(
        ...     dataset, "my_dataset_v1",
        ...     mode="semantic", threshold=0.9
        ... )
    """
    config = DedupConfig(
        mode=DedupMode.SEMANTIC if mode == "semantic" else DedupMode.EXACT,
        semantic_threshold=threshold,
        text_column=text_column,
        **kwargs,
    )

    cache_config = CacheConfig(cache_dir=cache_dir) if cache_dir else CacheConfig()

    pipeline = StaticDedupPipeline(config, cache_config)
    return pipeline.process_and_cache(
        dataset, dataset_name, force_reprocess=force, verbose=verbose
    )
