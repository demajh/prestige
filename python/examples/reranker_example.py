#!/usr/bin/env python3
"""
Example: Two-stage semantic deduplication with BGE reranker.

This example shows how to use prestige with a reranker for higher accuracy
semantic deduplication. The system first retrieves candidates with fast 
embeddings, then reranks them with a cross-encoder model.
"""

import tempfile
import shutil
from pathlib import Path

import prestige


def main():
    # Create temporary directory for the store
    temp_dir = Path(tempfile.mkdtemp(prefix="prestige_reranker_"))
    try:
        run_reranker_example(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_reranker_example(store_path: Path):
    print("=== BGE Reranker Example ===\n")
    
    # Configure store with reranker
    options = prestige.Options()
    options.dedup_mode = prestige.DedupMode.SEMANTIC
    
    # Basic semantic settings
    options.semantic_model_path = "./models/bge-small/model.onnx"
    options.semantic_model_type = prestige.SemanticModel.BGE_SMALL
    options.semantic_threshold = 0.75  # Lower threshold for initial retrieval
    
    # Enable reranker for higher accuracy
    options.semantic_reranker_enabled = True
    options.semantic_reranker_model_path = "./models/bge-reranker-v2-m3/model.onnx"
    options.semantic_reranker_top_k = 50  # Retrieve more candidates
    options.semantic_reranker_threshold = 0.8  # Higher threshold for final decision
    options.semantic_reranker_batch_size = 8  # Process in batches
    options.semantic_reranker_fallback = True  # Fall back if reranker fails
    
    print("Opening store with two-stage semantic deduplication:")
    print(f"  Embedding model: {options.semantic_model_path}")
    print(f"  Reranker model: {options.semantic_reranker_model_path}")
    print(f"  Initial threshold: {options.semantic_threshold}")
    print(f"  Reranker threshold: {options.semantic_reranker_threshold}")
    print()
    
    try:
        with prestige.Store.open(str(store_path), options) as store:
            # Store some similar texts that should be deduplicated
            texts = [
                ("news1", "Apple Inc. reported strong quarterly earnings today."),
                ("news2", "Apple posted robust financial results this quarter."),
                ("news3", "Tech giant Apple announced impressive Q3 earnings."),
                ("article1", "The weather forecast predicts rain tomorrow."),
                ("article2", "Meteorologists expect precipitation tomorrow."),
                ("doc1", "Python is a popular programming language."),
                ("doc2", "Python is widely used for software development."),
                ("unrelated", "The cat sat on the mat."),
            ]
            
            print("Storing texts with semantic deduplication...")
            for key, text in texts:
                store.put(key, text)
                print(f"  Stored: {key}")
            
            print()
            
            # Check deduplication results
            total_keys = store.count_keys()
            unique_objects = store.count_unique_values()
            
            print("Deduplication Results:")
            print(f"  Total keys: {total_keys}")
            print(f"  Unique objects: {unique_objects}")
            print(f"  Deduplication ratio: {total_keys / unique_objects:.2f}x")
            print()
            
            # Verify we can retrieve all texts
            print("Retrieved texts:")
            for key, original_text in texts:
                retrieved = store.get(key, decode=True)
                match_symbol = "✓" if retrieved == original_text else "✗"
                print(f"  {key}: {match_symbol}")
            print()
            
            # Show which keys map to the same objects (semantic duplicates)
            print("Object ID mapping (same ID = deduplicated):")
            object_groups = {}
            for key, _ in texts:
                obj_id = store.get_object_id(key)
                if obj_id not in object_groups:
                    object_groups[obj_id] = []
                object_groups[obj_id].append(key)
            
            for i, (obj_id, keys) in enumerate(object_groups.items(), 1):
                if len(keys) > 1:
                    print(f"  Group {i}: {', '.join(keys)} → {obj_id[:8]}...")
                else:
                    print(f"  Unique: {keys[0]} → {obj_id[:8]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: This example requires:")
        print("  1. BGE embedding model at ./models/bge-small/")
        print("  2. BGE reranker model at ./models/bge-reranker-v2-m3/")
        print("  3. Built with PRESTIGE_ENABLE_SEMANTIC=ON")


def download_models():
    """Helper to download required models (implementation not shown)."""
    print("To download models, run:")
    print("  python benchmarks/semantic_dedup/cli.py download bge-small")
    print("  # Download BGE reranker from Hugging Face")


if __name__ == "__main__":
    main()