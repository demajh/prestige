"""Deduplication store wrapper for tracking dedup decisions.

This module provides a wrapper around prestige.Store that tracks deduplication
decisions and provides metrics collection.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple
import tempfile
import shutil

import prestige

from .config import DedupConfig, DedupMode
from .metrics import DedupMetrics


class DedupStore:
    """Wrapper around prestige.Store that tracks deduplication decisions.

    This class wraps a prestige store and provides additional tracking of
    which items were deduplicated, enabling metrics collection and analysis.

    The key pattern for deduplication detection:
        1. store.put(key, text) - adds the item
        2. store.get_object_id(key) - gets the internal object ID
        3. If two items map to the same object ID, they were deduplicated

    Attributes:
        config: DedupConfig controlling behavior
        metrics: DedupMetrics collecting statistics

    Example:
        >>> config = DedupConfig(mode=DedupMode.SEMANTIC, semantic_threshold=0.9)
        >>> with DedupStore(config) as store:
        ...     is_dup, obj_id = store.check_and_add("item_0", "hello world")
        ...     is_dup, obj_id = store.check_and_add("item_1", "hello world!")
        ...     print(store.metrics.summary())
    """

    def __init__(self, config: DedupConfig):
        """Initialize the dedup store.

        Args:
            config: Configuration for deduplication behavior
        """
        self.config = config
        self.metrics = DedupMetrics()
        self._store: Optional[prestige.Store] = None
        self._temp_dir: Optional[Path] = None
        self._seen_object_ids: Set[bytes] = set()
        self._key_to_object_id: Dict[str, bytes] = {}
        self._is_open = False

    def open(self) -> "DedupStore":
        """Open or create the prestige store.

        Returns:
            Self for chaining

        Raises:
            RuntimeError: If store is already open
        """
        if self._is_open:
            raise RuntimeError("Store is already open")

        # Determine store path
        if self.config.store_path:
            store_path = str(self.config.store_path)
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="prestige_dedup_"))
            store_path = str(self._temp_dir / "store")

        # Configure options
        options = self._build_options()

        # Open the store
        self._store = prestige.open(store_path, options)
        self._is_open = True
        self.metrics.start()

        return self

    def _build_options(self) -> prestige.Options:
        """Build prestige Options from DedupConfig.

        Returns:
            Configured prestige.Options object

        Raises:
            RuntimeError: If semantic mode is requested but not available
        """
        options = prestige.Options()

        if self.config.mode == DedupMode.SEMANTIC:
            # Check if semantic support is available
            if not prestige.SEMANTIC_AVAILABLE:
                raise RuntimeError(
                    "Semantic mode requires prestige built with PRESTIGE_ENABLE_SEMANTIC=ON. "
                    "Please rebuild prestige with semantic support or use mode=EXACT."
                )

            options.dedup_mode = prestige.DedupMode.SEMANTIC
            options.semantic_threshold = self.config.semantic_threshold

            # Model configuration
            model_type = self._get_model_type()
            if model_type is not None:
                options.semantic_model_type = model_type

            if self.config.semantic_model_path:
                options.semantic_model_path = str(self.config.semantic_model_path)

            # Device configuration
            device = self._get_device()
            if device is not None:
                options.semantic_device = device

            options.semantic_num_threads = self.config.semantic_num_threads
            options.semantic_search_k = self.config.semantic_search_k

            # Reranker
            if self.config.enable_reranker:
                options.semantic_reranker_enabled = True
                options.semantic_reranker_threshold = self.config.reranker_threshold
                options.semantic_reranker_top_k = self.config.reranker_top_k
                options.semantic_reranker_batch_size = self.config.reranker_batch_size
                if self.config.reranker_model_path:
                    options.semantic_reranker_model_path = str(
                        self.config.reranker_model_path
                    )

            # RNN
            if self.config.enable_rnn:
                options.semantic_rnn_enabled = True
                if self.config.rnn_k > 0:
                    options.semantic_rnn_k = self.config.rnn_k

            # Margin gating
            if self.config.enable_margin_gating:
                options.semantic_margin_enabled = True
                options.semantic_margin_threshold = self.config.margin_threshold

        # For EXACT mode, we don't need to set dedup_mode - it defaults to exact
        # This allows the dataloaders to work with prestige builds that don't have semantic support

        return options

    def _get_model_type(self) -> Optional[Any]:
        """Map model type string to prestige.SemanticModel enum.

        Returns:
            SemanticModel enum value or None if not available
        """
        if not prestige.SEMANTIC_AVAILABLE:
            return None

        model_map = {
            "minilm": prestige.SemanticModel.MINILM,
            "bge-small": prestige.SemanticModel.BGE_SMALL,
            "bge-large": prestige.SemanticModel.BGE_LARGE,
            "e5-large": prestige.SemanticModel.E5_LARGE,
            "bge-m3": prestige.SemanticModel.BGE_M3,
            "nomic": prestige.SemanticModel.NOMIC_EMBED,
        }

        model_key = self.config.semantic_model_type.lower().replace("_", "-")
        return model_map.get(model_key)

    def _get_device(self) -> Optional[Any]:
        """Map device string to prestige.SemanticDevice enum.

        Returns:
            SemanticDevice enum value or None if not available
        """
        if not prestige.SEMANTIC_AVAILABLE:
            return None

        device_map = {
            "auto": prestige.SemanticDevice.AUTO,
            "cpu": prestige.SemanticDevice.CPU,
            "gpu": prestige.SemanticDevice.GPU,
        }

        return device_map.get(self.config.semantic_device.lower())

    def check_and_add(self, key: str, value: str) -> Tuple[bool, bytes]:
        """Check if value is a duplicate and add to store.

        This is the core deduplication operation. It adds the value to the store
        and checks if it was deduplicated (mapped to an existing object).

        Args:
            key: Unique key for this item
            value: Text value to check/add

        Returns:
            Tuple of (is_duplicate, object_id)
            - is_duplicate: True if value matched an existing item
            - object_id: The internal object ID assigned to this value

        Raises:
            RuntimeError: If store is not open
        """
        if not self._is_open or self._store is None:
            raise RuntimeError("Store is not open. Call open() first.")

        # Add to store
        self._store.put(key, value)
        object_id = self._store.get_object_id(key)

        # Check if we've seen this object ID before
        is_duplicate = object_id in self._seen_object_ids

        if not is_duplicate:
            self._seen_object_ids.add(object_id)

        # Track key to object ID mapping
        self._key_to_object_id[key] = object_id

        return is_duplicate, object_id

    def check_contamination(
        self, key: str, value: str
    ) -> Tuple[bool, Optional[bytes]]:
        """Check if value exists in store without adding it permanently.

        Used for contamination detection - checks if a value would deduplicate
        with existing data without polluting the index.

        Args:
            key: Temporary key for checking
            value: Text value to check

        Returns:
            Tuple of (is_contaminated, matching_object_id)
            - is_contaminated: True if value matches existing data
            - matching_object_id: The object ID if contaminated, None otherwise

        Raises:
            RuntimeError: If store is not open
        """
        if not self._is_open or self._store is None:
            raise RuntimeError("Store is not open. Call open() first.")

        # Use a temporary key
        temp_key = f"__contamination_check__{key}"

        try:
            # Add temporarily
            self._store.put(temp_key, value)
            object_id = self._store.get_object_id(temp_key)

            # Check if this object ID was already in the store
            is_contaminated = object_id in self._seen_object_ids

            # Clean up
            self._store.delete(temp_key)

            return is_contaminated, object_id if is_contaminated else None

        except Exception:
            # Try to clean up on error
            try:
                self._store.delete(temp_key)
            except Exception:
                pass
            return False, None

    def get_object_id(self, key: str) -> Optional[bytes]:
        """Get the object ID for a key.

        Args:
            key: Key to look up

        Returns:
            Object ID if key exists, None otherwise
        """
        return self._key_to_object_id.get(key)

    def is_duplicate_of(self, key1: str, key2: str) -> bool:
        """Check if two keys point to the same object.

        Args:
            key1: First key
            key2: Second key

        Returns:
            True if both keys map to the same object ID
        """
        obj1 = self._key_to_object_id.get(key1)
        obj2 = self._key_to_object_id.get(key2)

        if obj1 is None or obj2 is None:
            return False

        return obj1 == obj2

    def flush(self) -> None:
        """Flush pending writes to ensure durability."""
        if self._store is not None:
            self._store.flush()

    def get_health(self) -> Dict[str, Any]:
        """Get health statistics from the underlying store.

        Returns:
            Dictionary with store health statistics
        """
        if self._store is not None:
            return self._store.get_health()
        return {}

    def get_metrics(self) -> DedupMetrics:
        """Get current deduplication metrics.

        Updates metrics with store health stats before returning.

        Returns:
            DedupMetrics with current statistics
        """
        if self._store is not None:
            health = self._store.get_health()
            self.metrics.update_from_store_health(health)
        return self.metrics

    @property
    def unique_count(self) -> int:
        """Number of unique objects seen."""
        return len(self._seen_object_ids)

    @property
    def total_keys(self) -> int:
        """Total number of keys added."""
        return len(self._key_to_object_id)

    @property
    def dedup_ratio(self) -> float:
        """Current deduplication ratio."""
        if self.unique_count == 0:
            return 1.0
        return self.total_keys / self.unique_count

    def close(self) -> None:
        """Close the store and optionally clean up temporary files."""
        # Use 'is not None' instead of truthy check because prestige.Store
        # may have a __bool__ that returns False
        if self._store is not None:
            self.metrics.finish()
            # Update metrics one last time
            try:
                health = self._store.get_health()
                self.metrics.update_from_store_health(health)
            except Exception:
                pass

            self._store.close()
            self._store = None
            self._is_open = False

        # Clean up temporary directory if not persisting
        if self._temp_dir is not None and not self.config.persist_store:
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_dir = None

    def reset_tracking(self) -> None:
        """Reset internal tracking state.

        Useful when reusing the store for a new deduplication pass.
        Does not affect the underlying store data.
        """
        self._seen_object_ids.clear()
        self._key_to_object_id.clear()
        self.metrics = DedupMetrics()

    def __enter__(self) -> "DedupStore":
        """Context manager entry."""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        status = "open" if self._is_open else "closed"
        return (
            f"DedupStore(mode={self.config.mode.value}, "
            f"keys={self.total_keys}, unique={self.unique_count}, "
            f"status={status})"
        )


def create_dedup_store(
    mode: str = "exact",
    threshold: float = 0.85,
    store_path: Optional[Path] = None,
    **kwargs,
) -> DedupStore:
    """Convenience function to create a DedupStore.

    Args:
        mode: "exact" or "semantic"
        threshold: Semantic similarity threshold (0.0-1.0)
        store_path: Optional custom store path
        **kwargs: Additional DedupConfig options

    Returns:
        Configured DedupStore instance (not yet open)

    Example:
        >>> store = create_dedup_store(mode="semantic", threshold=0.9)
        >>> with store:
        ...     is_dup, _ = store.check_and_add("k1", "hello")
    """
    config = DedupConfig(
        mode=DedupMode.SEMANTIC if mode == "semantic" else DedupMode.EXACT,
        semantic_threshold=threshold,
        store_path=store_path,
        **kwargs,
    )
    return DedupStore(config)
