"""Contamination detection for train/test leakage and cross-dataset deduplication.

This module provides tools for detecting data contamination, including:
- Train/test leakage detection
- Cross-dataset deduplication
- Reference index building for contamination checks
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .config import DedupConfig, DedupMode, CrossDatasetConfig
from .dedup_store import DedupStore
from .metrics import DedupMetrics
from .utils import extract_text, make_key

# Try to import HuggingFace datasets
try:
    from datasets import Dataset as HFDataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    HFDataset = None


class ContaminationDetector:
    """Detects train/test leakage and cross-dataset contamination.

    This class builds an index from a reference dataset (e.g., test set) and
    then checks a target dataset (e.g., training set) for contamination.

    Example:
        >>> detector = ContaminationDetector(config)
        >>> detector.build_reference_index(test_data)
        >>> contaminated, metrics = detector.check_contamination(train_data)
        >>> print(f"Found {len(contaminated)} contaminated examples")
    """

    def __init__(
        self,
        config: DedupConfig,
        cross_config: Optional[CrossDatasetConfig] = None,
    ):
        """Initialize the contamination detector.

        Args:
            config: DedupConfig for deduplication settings
            cross_config: CrossDatasetConfig for contamination thresholds
        """
        self.config = config
        self.cross_config = cross_config or CrossDatasetConfig()
        self._reference_store: Optional[DedupStore] = None
        self._metrics = DedupMetrics()
        self._reference_count = 0

    def build_reference_index(
        self,
        reference_data: Union[Sequence[Any], "HFDataset"],
        reference_name: str = "reference",
        text_column: Optional[str] = None,
        verbose: bool = False,
    ) -> int:
        """Build an index from the reference dataset.

        Call this first with the dataset you want to check against
        (e.g., your test set).

        Args:
            reference_data: Reference dataset (e.g., test set)
            reference_name: Name for the reference (used in store path)
            text_column: Column containing text (overrides config)
            verbose: Print progress information

        Returns:
            Number of items indexed

        Example:
            >>> detector.build_reference_index(test_dataset, "test_set")
            1000
        """
        text_col = text_column or self.config.text_column

        # Determine store path
        if self.cross_config.reference_store_path:
            store_path = self.cross_config.reference_store_path
        else:
            store_path = (
                Path.home()
                / ".cache"
                / "prestige"
                / "contamination"
                / reference_name
            )

        # Ensure parent directory exists
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Create reference config with higher threshold for contamination
        ref_config = DedupConfig(
            mode=self.config.mode,
            semantic_threshold=self.cross_config.contamination_threshold,
            semantic_model_type=self.config.semantic_model_type,
            semantic_model_path=self.config.semantic_model_path,
            semantic_device=self.config.semantic_device,
            store_path=store_path,
            persist_store=True,  # Keep for checking
            text_column=text_col,
            batch_size=self.config.batch_size,
        )

        # Close existing store if any
        if self._reference_store is not None:
            self._reference_store.close()

        self._reference_store = DedupStore(ref_config)
        self._reference_store.open()

        # Index all reference items
        total = len(reference_data)
        for idx in range(total):
            item = reference_data[idx]
            text = extract_text(item, text_col)
            key = f"ref_{reference_name}_{idx}"

            self._reference_store.check_and_add(key, text)

            if (idx + 1) % self.config.batch_size == 0:
                self._reference_store.flush()
                if verbose:
                    print(f"Indexed {idx + 1}/{total} reference items")

        self._reference_store.flush()
        self._reference_count = total

        if verbose:
            print(f"Built reference index with {total} items")

        return total

    def check_contamination(
        self,
        target_data: Union[Sequence[Any], "HFDataset"],
        text_column: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[List[int], DedupMetrics]:
        """Check target dataset for contamination against the reference.

        Args:
            target_data: Dataset to check (e.g., training set)
            text_column: Column containing text (overrides config)
            verbose: Print progress information

        Returns:
            Tuple of (contaminated_indices, metrics)
            - contaminated_indices: List of indices that are contaminated
            - metrics: DedupMetrics with contamination statistics

        Raises:
            RuntimeError: If build_reference_index was not called first

        Example:
            >>> contaminated, metrics = detector.check_contamination(train_data)
            >>> print(f"Contamination rate: {metrics.contamination_rate:.2%}")
        """
        if self._reference_store is None:
            raise RuntimeError("Must call build_reference_index first")

        text_col = text_column or self.config.text_column

        self._metrics = DedupMetrics()
        self._metrics.start()

        contaminated_indices = []
        total = len(target_data)

        for idx in range(total):
            item = target_data[idx]
            text = extract_text(item, text_col)

            is_contaminated, _ = self._reference_store.check_contamination(
                f"check_{idx}", text
            )

            self._metrics.total_seen += 1

            if is_contaminated:
                self._metrics.record_contamination(idx)
                contaminated_indices.append(idx)

            if verbose and (idx + 1) % self.config.batch_size == 0:
                print(
                    f"Checked {idx + 1}/{total}: "
                    f"{len(contaminated_indices)} contaminated"
                )

        self._metrics.finish()

        if verbose:
            print(
                f"Found {len(contaminated_indices)} contaminated items "
                f"({self._metrics.contamination_rate:.2%})"
            )

        return contaminated_indices, self._metrics

    def filter_contaminated(
        self,
        target_data: Union[Sequence[Any], "HFDataset"],
        text_column: Optional[str] = None,
        verbose: bool = False,
    ) -> Union[List[Any], "HFDataset"]:
        """Remove contaminated samples from the target dataset.

        Args:
            target_data: Dataset to filter
            text_column: Column containing text
            verbose: Print progress

        Returns:
            Filtered dataset with contaminated samples removed
        """
        contaminated_indices, _ = self.check_contamination(
            target_data, text_column, verbose
        )
        contaminated_set = set(contaminated_indices)

        # Handle different data types
        if HF_AVAILABLE and isinstance(target_data, HFDataset):
            clean_indices = [
                idx for idx in range(len(target_data)) if idx not in contaminated_set
            ]
            return target_data.select(clean_indices)
        else:
            return [
                item
                for idx, item in enumerate(target_data)
                if idx not in contaminated_set
            ]

    def get_metrics(self) -> DedupMetrics:
        """Get metrics from last contamination check.

        Returns:
            DedupMetrics with contamination statistics
        """
        return self._metrics

    @property
    def reference_count(self) -> int:
        """Number of items in the reference index."""
        return self._reference_count

    def close(self) -> None:
        """Close the reference store and clean up."""
        if self._reference_store:
            self._reference_store.close()
            self._reference_store = None

    def __enter__(self) -> "ContaminationDetector":
        return self

    def __exit__(self, *args) -> None:
        self.close()


def detect_train_test_leakage(
    train_data: Union[Sequence[Any], "HFDataset"],
    test_data: Union[Sequence[Any], "HFDataset"],
    mode: str = "semantic",
    threshold: float = 0.95,
    text_column: str = "text",
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """One-liner to detect train/test contamination.

    Checks if any examples in the training set are semantically similar
    to examples in the test set. Uses a higher threshold by default
    since contamination detection requires high precision.

    Args:
        train_data: Training dataset to check
        test_data: Test dataset (reference)
        mode: "exact" or "semantic"
        threshold: Similarity threshold (0.95 default for contamination)
        text_column: Column containing text
        verbose: Print progress
        **kwargs: Additional DedupConfig options

    Returns:
        Dictionary with:
        - contaminated_train_indices: List of contaminated training indices
        - contaminated_count: Number of contaminated samples
        - contamination_rate: Fraction of training data contaminated
        - reference_count: Number of test examples
        - metrics: Full DedupMetrics object

    Example:
        >>> results = detect_train_test_leakage(train_ds, test_ds)
        >>> if results["contaminated_count"] > 0:
        ...     print(f"WARNING: {results['contamination_rate']:.2%} contamination!")
    """
    config = DedupConfig(
        mode=DedupMode.SEMANTIC if mode == "semantic" else DedupMode.EXACT,
        semantic_threshold=threshold,
        text_column=text_column,
        **kwargs,
    )
    cross_config = CrossDatasetConfig(contamination_threshold=threshold)

    detector = ContaminationDetector(config, cross_config)

    try:
        # Build index from test set
        reference_count = detector.build_reference_index(
            test_data, "test_set", verbose=verbose
        )

        # Check training data
        contaminated, metrics = detector.check_contamination(train_data, verbose=verbose)

        return {
            "contaminated_train_indices": contaminated,
            "contaminated_count": len(contaminated),
            "contamination_rate": (
                len(contaminated) / len(train_data) if train_data else 0
            ),
            "reference_count": reference_count,
            "metrics": metrics,
        }
    finally:
        detector.close()


def filter_train_test_leakage(
    train_data: Union[Sequence[Any], "HFDataset"],
    test_data: Union[Sequence[Any], "HFDataset"],
    mode: str = "semantic",
    threshold: float = 0.95,
    text_column: str = "text",
    verbose: bool = False,
    **kwargs,
) -> Union[List[Any], "HFDataset"]:
    """Remove contaminated examples from training data.

    Convenience function that detects and removes contaminated examples.

    Args:
        train_data: Training dataset to filter
        test_data: Test dataset (reference)
        mode: "exact" or "semantic"
        threshold: Similarity threshold
        text_column: Column containing text
        verbose: Print progress
        **kwargs: Additional DedupConfig options

    Returns:
        Filtered training dataset

    Example:
        >>> clean_train = filter_train_test_leakage(train_ds, test_ds)
        >>> print(f"Removed {len(train_ds) - len(clean_train)} contaminated examples")
    """
    config = DedupConfig(
        mode=DedupMode.SEMANTIC if mode == "semantic" else DedupMode.EXACT,
        semantic_threshold=threshold,
        text_column=text_column,
        **kwargs,
    )
    cross_config = CrossDatasetConfig(contamination_threshold=threshold)

    with ContaminationDetector(config, cross_config) as detector:
        detector.build_reference_index(test_data, "test_set", verbose=verbose)
        return detector.filter_contaminated(train_data, verbose=verbose)


class CrossDatasetDeduplicator:
    """Deduplicate across multiple datasets.

    Useful for combining datasets while removing duplicates that appear
    across different sources.

    Example:
        >>> deduplicator = CrossDatasetDeduplicator(config)
        >>> deduplicator.add_dataset(dataset1, "wiki")
        >>> deduplicator.add_dataset(dataset2, "books")  # Deduped against wiki
        >>> combined = deduplicator.get_combined()
    """

    def __init__(self, config: DedupConfig):
        """Initialize cross-dataset deduplicator.

        Args:
            config: DedupConfig for deduplication settings
        """
        self.config = config
        self._store: Optional[DedupStore] = None
        self._datasets: Dict[str, List[int]] = {}  # name -> valid indices
        self._metrics = DedupMetrics()

    def _ensure_store(self) -> DedupStore:
        """Ensure the dedup store is open."""
        if self._store is None:
            self._store = DedupStore(self.config)
            self._store.open()
            self._metrics.start()
        return self._store

    def add_dataset(
        self,
        data: Union[Sequence[Any], "HFDataset"],
        name: str,
        text_column: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[int, int]:
        """Add a dataset, deduplicating against all previously added data.

        Args:
            data: Dataset to add
            name: Unique name for this dataset
            text_column: Column containing text
            verbose: Print progress

        Returns:
            Tuple of (kept_count, removed_count)

        Example:
            >>> kept, removed = deduplicator.add_dataset(wiki_data, "wiki")
            >>> print(f"Wiki: kept {kept}, removed {removed} cross-dataset dups")
        """
        store = self._ensure_store()
        text_col = text_column or self.config.text_column

        valid_indices = []
        total = len(data)

        for idx in range(total):
            item = data[idx]
            text = extract_text(item, text_col)
            key = f"{name}_{idx}"

            is_duplicate, _ = store.check_and_add(key, text)
            self._metrics.record_item(is_duplicate, index=idx, key=key)

            if not is_duplicate:
                valid_indices.append(idx)

            if (idx + 1) % self.config.batch_size == 0:
                store.flush()
                if verbose:
                    print(f"{name}: {idx + 1}/{total}, kept {len(valid_indices)}")

        store.flush()
        self._datasets[name] = valid_indices

        kept = len(valid_indices)
        removed = total - kept

        if verbose:
            print(f"{name}: kept {kept}, removed {removed}")

        return kept, removed

    def get_valid_indices(self, name: str) -> List[int]:
        """Get valid (non-duplicate) indices for a dataset.

        Args:
            name: Dataset name

        Returns:
            List of valid indices
        """
        return self._datasets.get(name, [])

    def filter_dataset(
        self, data: Union[Sequence[Any], "HFDataset"], name: str
    ) -> Union[List[Any], "HFDataset"]:
        """Get filtered version of a dataset.

        Args:
            data: Original dataset
            name: Dataset name (must have been added first)

        Returns:
            Filtered dataset
        """
        valid_indices = self._datasets.get(name, [])

        if HF_AVAILABLE and isinstance(data, HFDataset):
            return data.select(valid_indices)
        else:
            return [data[idx] for idx in valid_indices]

    def get_metrics(self) -> DedupMetrics:
        """Get combined metrics."""
        if self._store:
            self._metrics.update_from_store_health(self._store.get_health())
        return self._metrics

    def close(self) -> None:
        """Close the store."""
        if self._store:
            self._metrics.finish()
            self._store.close()
            self._store = None

    def __enter__(self) -> "CrossDatasetDeduplicator":
        return self

    def __exit__(self, *args) -> None:
        self.close()
