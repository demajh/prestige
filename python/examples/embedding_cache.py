#!/usr/bin/env python3
"""Embedding cache example for Prestige Python bindings.

This example shows how to use Prestige as a cache for expensive
embedding computations (e.g., from OpenAI, Cohere, local models).

The cache automatically deduplicates identical text, so if you embed
the same text multiple times, it only computes the embedding once.
"""

import prestige
import tempfile
import shutil
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import List, Optional


class EmbeddingCache:
    """Simple embedding cache wrapper around Prestige.

    Features:
    - Automatic deduplication of identical texts
    - TTL support for cache expiration
    - Binary storage of embedding vectors
    - Hit/miss statistics
    """

    def __init__(self, db_path: str, ttl_seconds: int = 7 * 24 * 3600):
        """Initialize the embedding cache.

        Args:
            db_path: Path to the database directory
            ttl_seconds: Time-to-live for cache entries (default: 7 days)
        """
        options = prestige.Options()
        options.default_ttl_seconds = ttl_seconds
        self.store = prestige.open(db_path, options)
        self.hits = 0
        self.misses = 0

    def get_embedding(
        self,
        text: str,
        model: str = "default",
        compute_fn=None,
    ) -> Optional[List[float]]:
        """Get embedding from cache or compute it.

        Args:
            text: Input text to embed
            model: Model identifier (for cache key namespacing)
            compute_fn: Function to compute embedding if not cached
                       Should take (text: str) and return List[float]

        Returns:
            Embedding vector as list of floats, or None if not cached
            and no compute_fn provided
        """
        # Create cache key: model:hash(text)
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"emb:{model}:{text_hash}"

        try:
            # Try to get from cache
            cached_bytes = self.store.get(cache_key)
            self.hits += 1

            # Deserialize from binary format
            embedding = self._deserialize_embedding(cached_bytes)
            return embedding

        except prestige.NotFoundError:
            self.misses += 1

            # Compute if function provided
            if compute_fn is not None:
                embedding = compute_fn(text)

                # Cache the result
                embedding_bytes = self._serialize_embedding(embedding)
                self.store.put(cache_key, embedding_bytes)

                return embedding

            return None

    def put_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = "default",
    ):
        """Manually put an embedding into the cache.

        Args:
            text: Input text
            embedding: Embedding vector
            model: Model identifier
        """
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        cache_key = f"emb:{model}:{text_hash}"

        embedding_bytes = self._serialize_embedding(embedding)
        self.store.put(cache_key, embedding_bytes)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        health = self.store.get_health()
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_keys": health["total_keys"],
            "unique_embeddings": health["total_objects"],
            "dedup_ratio": health["dedup_ratio"],
            "storage_bytes": health["total_bytes"],
        }

    def clear_expired(self) -> int:
        """Remove expired cache entries.

        Returns:
            Number of entries removed
        """
        return self.store.prune()

    def close(self):
        """Close the cache."""
        if hasattr(self, "store"):
            self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def _serialize_embedding(embedding: List[float]) -> bytes:
        """Serialize embedding to binary format.

        Format: [dimensions:4bytes][values:4bytes each]
        """
        dims = len(embedding)
        # Pack as: dimension count (int) + float values
        packed = struct.pack(f"<I{dims}f", dims, *embedding)
        return packed

    @staticmethod
    def _deserialize_embedding(data: bytes) -> List[float]:
        """Deserialize embedding from binary format."""
        # Unpack dimension count
        dims = struct.unpack("<I", data[:4])[0]
        # Unpack float values
        values = struct.unpack(f"<{dims}f", data[4:])
        return list(values)


def mock_embedding_function(text: str) -> List[float]:
    """Mock embedding function that simulates expensive computation.

    In reality, this would call OpenAI, Cohere, or a local model.
    """
    # Simulate some computation time
    time.sleep(0.01)

    # Generate a deterministic but pseudo-random embedding
    # (In real use, this would be the actual model output)
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    embedding = [(hash_val >> i) % 100 / 100.0 for i in range(0, 384, 4)]
    return embedding[:96]  # 96-dimensional embedding


def main():
    print("=== Embedding Cache Example ===\n")

    # Create temporary database
    db_path = Path(tempfile.mkdtemp()) / "embeddings_cache"
    print(f"Creating cache at: {db_path}\n")

    try:
        with EmbeddingCache(str(db_path), ttl_seconds=3600) as cache:
            print("=== Computing Embeddings (First Time) ===")

            texts = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is transforming technology",
                "Python is a versatile programming language",
                "The quick brown fox jumps over the lazy dog",  # Duplicate!
                "Data science requires statistical knowledge",
                "Machine learning is transforming technology",  # Duplicate!
            ]

            start_time = time.time()
            embeddings = []
            for text in texts:
                emb = cache.get_embedding(text, model="test-model", compute_fn=mock_embedding_function)
                embeddings.append(emb)
                print(f"  ✓ Embedded: '{text[:50]}...'")

            first_run_time = time.time() - start_time
            print(f"\nFirst run took: {first_run_time:.3f}s")

            print("\n=== Cache Statistics (After First Run) ===")
            stats = cache.get_stats()
            print(f"  Hits: {stats['hits']}")
            print(f"  Misses: {stats['misses']}")
            print(f"  Hit rate: {stats['hit_rate']:.1%}")
            print(f"  Total keys: {stats['total_keys']}")
            print(f"  Unique embeddings: {stats['unique_embeddings']}")
            print(f"  Deduplication ratio: {stats['dedup_ratio']:.1f}x")
            print(f"  Storage: {stats['storage_bytes'] / 1024:.1f} KB")

            print("\n=== Retrieving Embeddings (From Cache) ===")

            # Reset stats to show cache hits
            cache.hits = 0
            cache.misses = 0

            start_time = time.time()
            cached_embeddings = []
            for text in texts:
                emb = cache.get_embedding(text, model="test-model")
                cached_embeddings.append(emb)
                print(f"  ✓ Retrieved: '{text[:50]}...'")

            second_run_time = time.time() - start_time
            print(f"\nSecond run took: {second_run_time:.3f}s")
            print(f"Speedup: {first_run_time / second_run_time:.1f}x faster")

            print("\n=== Cache Statistics (After Second Run) ===")
            stats = cache.get_stats()
            print(f"  Hits: {stats['hits']}")
            print(f"  Misses: {stats['misses']}")
            print(f"  Hit rate: {stats['hit_rate']:.1%}")

            # Verify embeddings are identical
            print("\n=== Verification ===")
            all_match = all(
                e1 == e2 for e1, e2 in zip(embeddings, cached_embeddings)
            )
            print(f"  ✓ Embeddings match: {all_match}")

            # Show deduplication benefit
            print("\n=== Deduplication Benefit ===")
            print(f"  Texts processed: {len(texts)}")
            print(f"  Unique texts: {stats['unique_embeddings']}")
            print(f"  Storage savings: {stats['dedup_ratio']:.1f}x")
            print(
                f"  (Stored {len(texts)} texts but only computed"
                f" {stats['unique_embeddings']} unique embeddings)"
            )

            print("\n=== Batch Processing Example ===")

            # Simulate batch processing with many duplicates
            batch_texts = []
            for i in range(10):
                batch_texts.extend(texts)  # 60 texts (10x6), but only 4 unique

            cache.hits = 0
            cache.misses = 0

            start_time = time.time()
            for text in batch_texts:
                cache.get_embedding(text, model="test-model", compute_fn=mock_embedding_function)

            batch_time = time.time() - start_time

            stats = cache.get_stats()
            print(f"  Processed {len(batch_texts)} texts in {batch_time:.3f}s")
            print(f"  Cache hits: {cache.hits}")
            print(f"  Cache misses: {cache.misses}")
            print(f"  Hit rate: {cache.hits / len(batch_texts):.1%}")

    finally:
        # Clean up
        if db_path.exists():
            shutil.rmtree(db_path.parent)
            print(f"\n✓ Cleaned up cache at {db_path}")


if __name__ == "__main__":
    main()
