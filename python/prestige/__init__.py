"""Prestige: Content-deduplicated key-value store.

Prestige is a high-performance key-value store with automatic content
deduplication. Identical values are stored only once, reducing storage
requirements for data with redundancy.

Basic usage:
    >>> import prestige
    >>> with prestige.open("/tmp/mydb") as store:
    ...     store.put("key", "value")
    ...     value = store.get("key", decode=True)
    ...     print(value)
    'value'

For more control, use Options:
    >>> options = prestige.Options()
    >>> options.default_ttl_seconds = 3600  # 1 hour TTL
    >>> with prestige.open("/tmp/mydb", options) as store:
    ...     store.put("key", "value")
"""

from ._prestige import (
    # Core
    Store,
    Options,
    HealthStats,
    # Enums
    DedupMode,
    NormalizationMode,
    # Exceptions
    PrestigeError,
    NotFoundError,
    InvalidArgumentError,
    IOError,
    CorruptionError,
    BusyError,
    TimedOutError,
    # Feature flags
    __version__,
    SEMANTIC_AVAILABLE,
    SERVER_AVAILABLE,
)

# Conditional imports for semantic features
if SEMANTIC_AVAILABLE:
    from ._prestige import SemanticModel, SemanticIndexType


def open(path: str, options: Options = None, **kwargs) -> Store:
    """Open or create a prestige store.

    This is a convenience function that creates an Options object
    from keyword arguments and opens a store.

    Args:
        path: Database path (directory will be created if needed)
        options: Pre-configured Options object (optional)
        **kwargs: Options attributes to set (e.g., default_ttl_seconds=3600)

    Returns:
        Store instance (usable as context manager)

    Examples:
        # Simple usage
        with prestige.open("/tmp/mydb") as store:
            store.put("key", "value")

        # With options
        with prestige.open("/tmp/mydb", default_ttl_seconds=3600) as store:
            store.put("key", "value")

        # With pre-configured options
        options = prestige.Options()
        options.max_store_bytes = 10 * 1024 * 1024 * 1024  # 10GB
        with prestige.open("/tmp/mydb", options) as store:
            store.put("key", "value")
    """
    if options is None:
        options = Options()

    for key, value in kwargs.items():
        if hasattr(options, key):
            setattr(options, key, value)
        else:
            raise ValueError(f"Unknown option: {key}")

    return Store.open(path, options)


__all__ = [
    # Factory function
    "open",
    # Core classes
    "Store",
    "Options",
    "HealthStats",
    # Enums
    "DedupMode",
    "NormalizationMode",
    # Exceptions
    "PrestigeError",
    "NotFoundError",
    "InvalidArgumentError",
    "IOError",
    "CorruptionError",
    "BusyError",
    "TimedOutError",
    # Feature flags
    "__version__",
    "SEMANTIC_AVAILABLE",
    "SERVER_AVAILABLE",
]

# Add semantic types to __all__ if available
if SEMANTIC_AVAILABLE:
    __all__.extend(["SemanticModel", "SemanticIndexType"])
