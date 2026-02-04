"""Metrics collection for deduplication operations.

This module provides dataclasses for tracking deduplication progress,
results, and performance statistics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import time


@dataclass
class DedupMetrics:
    """Metrics for tracking deduplication progress and results.

    This class collects statistics during deduplication operations including
    counts, timing, and optionally detailed information about removed items.

    Attributes:
        total_seen: Total number of items processed
        unique_kept: Number of unique items kept after deduplication
        duplicates_removed: Number of duplicate items removed
        start_time: Timestamp when processing started
        end_time: Timestamp when processing finished
        batch_times: List of per-batch processing times
        removed_indices: Indices of removed items (when tracking enabled)
        removed_keys: Keys of removed items (when tracking enabled)
        contaminated_count: Number of contaminated items found
        contaminated_indices: Indices of contaminated items

    Example:
        >>> metrics = DedupMetrics()
        >>> metrics.start()
        >>> for item in data:
        ...     is_dup = check_duplicate(item)
        ...     metrics.record_item(is_dup, index=i)
        >>> metrics.finish()
        >>> print(f"Removed {metrics.removal_rate:.1%} duplicates")
    """

    # Counts
    total_seen: int = 0
    unique_kept: int = 0
    duplicates_removed: int = 0

    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # Per-batch tracking
    batch_times: List[float] = field(default_factory=list)

    # Removed items (optional, for debugging/analysis)
    removed_indices: List[int] = field(default_factory=list)
    removed_keys: List[str] = field(default_factory=list)

    # Contamination detection
    contaminated_count: int = 0
    contaminated_indices: List[int] = field(default_factory=list)

    # Store-level metrics (populated from prestige health stats)
    storage_bytes: int = 0
    store_total_keys: int = 0
    store_unique_objects: int = 0

    @property
    def dedup_ratio(self) -> float:
        """Ratio of total items to unique items.

        A ratio of 2.0 means half the items were duplicates.
        A ratio of 1.0 means no duplicates were found.
        """
        if self.unique_kept == 0:
            return 1.0
        return self.total_seen / self.unique_kept

    @property
    def removal_rate(self) -> float:
        """Fraction of items removed as duplicates (0.0 to 1.0)."""
        if self.total_seen == 0:
            return 0.0
        return self.duplicates_removed / self.total_seen

    @property
    def keep_rate(self) -> float:
        """Fraction of items kept (1.0 - removal_rate)."""
        return 1.0 - self.removal_rate

    @property
    def elapsed_seconds(self) -> float:
        """Total processing time in seconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return end - self.start_time

    @property
    def items_per_second(self) -> float:
        """Processing throughput (items/sec)."""
        elapsed = self.elapsed_seconds
        if elapsed <= 0:
            return 0.0
        return self.total_seen / elapsed

    @property
    def contamination_rate(self) -> float:
        """Fraction of items that are contaminated."""
        if self.total_seen == 0:
            return 0.0
        return self.contaminated_count / self.total_seen

    def start(self) -> "DedupMetrics":
        """Mark start of processing. Returns self for chaining."""
        self.start_time = time.time()
        return self

    def finish(self) -> "DedupMetrics":
        """Mark end of processing. Returns self for chaining."""
        self.end_time = time.time()
        return self

    def record_item(
        self,
        is_duplicate: bool,
        index: Optional[int] = None,
        key: Optional[str] = None,
    ) -> None:
        """Record processing of a single item.

        Args:
            is_duplicate: Whether the item was identified as a duplicate
            index: Optional index of the item (for tracking removed items)
            key: Optional key of the item (for tracking removed items)
        """
        self.total_seen += 1
        if is_duplicate:
            self.duplicates_removed += 1
            if index is not None:
                self.removed_indices.append(index)
            if key is not None:
                self.removed_keys.append(key)
        else:
            self.unique_kept += 1

    def record_contamination(self, index: int, key: Optional[str] = None) -> None:
        """Record a contamination detection.

        Args:
            index: Index of the contaminated item
            key: Optional key of the contaminated item
        """
        self.contaminated_count += 1
        self.contaminated_indices.append(index)

    def record_batch_time(self, elapsed: float) -> None:
        """Record time taken for a batch.

        Args:
            elapsed: Time in seconds for the batch
        """
        self.batch_times.append(elapsed)

    def update_from_store_health(self, health: Dict[str, Any]) -> None:
        """Update metrics from prestige store health stats.

        Args:
            health: Dictionary from store.get_health()
        """
        self.storage_bytes = health.get("total_bytes", 0)
        self.store_total_keys = health.get("total_keys", 0)
        self.store_unique_objects = health.get("total_objects", 0)

    def merge(self, other: "DedupMetrics") -> "DedupMetrics":
        """Merge another metrics instance into this one.

        Useful for combining metrics from multiple workers.

        Args:
            other: Another DedupMetrics instance to merge

        Returns:
            Self for chaining
        """
        self.total_seen += other.total_seen
        self.unique_kept += other.unique_kept
        self.duplicates_removed += other.duplicates_removed
        self.contaminated_count += other.contaminated_count

        self.removed_indices.extend(other.removed_indices)
        self.removed_keys.extend(other.removed_keys)
        self.contaminated_indices.extend(other.contaminated_indices)
        self.batch_times.extend(other.batch_times)

        self.storage_bytes = max(self.storage_bytes, other.storage_bytes)
        self.store_total_keys = max(self.store_total_keys, other.store_total_keys)
        self.store_unique_objects = max(
            self.store_unique_objects, other.store_unique_objects
        )

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization.

        Returns:
            Dictionary containing all metrics
        """
        return {
            # Counts
            "total_seen": self.total_seen,
            "unique_kept": self.unique_kept,
            "duplicates_removed": self.duplicates_removed,
            # Rates
            "dedup_ratio": self.dedup_ratio,
            "removal_rate": self.removal_rate,
            "keep_rate": self.keep_rate,
            # Timing
            "elapsed_seconds": self.elapsed_seconds,
            "items_per_second": self.items_per_second,
            # Contamination
            "contaminated_count": self.contaminated_count,
            "contamination_rate": self.contamination_rate,
            # Storage
            "storage_bytes": self.storage_bytes,
            "store_total_keys": self.store_total_keys,
            "store_unique_objects": self.store_unique_objects,
        }

    def summary(self) -> str:
        """Generate a human-readable summary string.

        Returns:
            Formatted summary of metrics
        """
        lines = [
            f"Deduplication Results:",
            f"  Total processed: {self.total_seen:,}",
            f"  Unique kept:     {self.unique_kept:,}",
            f"  Duplicates:      {self.duplicates_removed:,} ({self.removal_rate:.1%})",
            f"  Dedup ratio:     {self.dedup_ratio:.2f}x",
        ]

        if self.elapsed_seconds > 0:
            lines.append(f"  Throughput:      {self.items_per_second:,.0f} items/sec")
            lines.append(f"  Elapsed time:    {self.elapsed_seconds:.1f}s")

        if self.contaminated_count > 0:
            lines.append(
                f"  Contaminated:    {self.contaminated_count:,} ({self.contamination_rate:.1%})"
            )

        if self.storage_bytes > 0:
            mb = self.storage_bytes / (1024 * 1024)
            lines.append(f"  Storage used:    {mb:.1f} MB")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DedupMetrics(total={self.total_seen}, unique={self.unique_kept}, "
            f"removed={self.duplicates_removed}, ratio={self.dedup_ratio:.2f}x)"
        )
