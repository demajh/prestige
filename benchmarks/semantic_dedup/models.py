"""Model downloader for semantic deduplication benchmarks."""

import hashlib
import json
from pathlib import Path
from typing import Optional, Dict
import urllib.request
import urllib.error


# Model registry with HuggingFace URLs and checksums
MODEL_REGISTRY = {
    "bge-small": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "BAAI/bge-small-en-v1.5",
        "model_url": "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt",
        "model_size_mb": 133,
        "description": "BGE-small English v1.5 (384 dims)",
        "dimensions": 384,
    },
    "bge-large": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "BAAI/bge-large-en-v1.5",
        "model_url": "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/vocab.txt",
        "model_size_mb": 1340,
        "description": "BGE-large English v1.5 (1024 dims)",
        "dimensions": 1024,
    },
    "bge-m3": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "BAAI/bge-m3",
        "model_url": "https://huggingface.co/BAAI/bge-m3/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/BAAI/bge-m3/resolve/main/vocab.txt",
        "model_size_mb": 2270,
        "description": "BGE-M3 multi-lingual (1024 dims)",
        "dimensions": 1024,
    },
    "e5-large": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "intfloat/e5-large-v2",
        "model_url": "https://huggingface.co/intfloat/e5-large-v2/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/intfloat/e5-large-v2/resolve/main/vocab.txt",
        "model_size_mb": 1340,
        "description": "E5-large-v2 (1024 dims)",
        "dimensions": 1024,
    },
    "nomic-embed": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "nomic-ai/nomic-embed-text-v1.5",
        "model_url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/vocab.txt",
        "model_size_mb": 548,
        "description": "Nomic-embed-text v1.5 (768 dims)",
        "dimensions": 768,
    },
    "minilm": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "model_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
        "model_size_mb": 90,
        "description": "MiniLM-L6-v2 (384 dims)",
        "dimensions": 384,
    },
    "bge-reranker-base": {
        "model_file": "model.onnx",
        "vocab_file": "vocab.txt",
        "hf_repo": "BAAI/bge-reranker-base",
        "model_url": "https://huggingface.co/BAAI/bge-reranker-base/resolve/main/onnx/model.onnx",
        "vocab_url": "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt",  # Use compatible vocab
        "model_size_mb": 279,
        "description": "BGE-reranker-base cross-encoder",
        "dimensions": 0,  # Cross-encoder, no embedding output
    },
}


class ModelDownloader:
    """Downloads and caches embedding models."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize model downloader.

        Args:
            cache_dir: Directory for caching models (default: ~/.cache/prestige/models)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "prestige" / "models"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_model(self, model_name: str, force: bool = False) -> Path:
        """Download a model and return path to ONNX file.

        Args:
            model_name: Name of model (e.g., "bge-small")
            force: Force re-download even if cached

        Returns:
            Path to ONNX model file

        Raises:
            ValueError: If model name not found
            RuntimeError: If download fails
        """
        if model_name not in MODEL_REGISTRY:
            available = ", ".join(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available models: {available}"
            )

        config = MODEL_REGISTRY[model_name]

        # Create model directory
        model_dir = self.cache_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / config["model_file"]
        vocab_path = model_dir / config["vocab_file"]

        # Check if already downloaded
        if model_path.exists() and vocab_path.exists() and not force:
            print(f"Model {model_name} already cached at {model_path}")
            return model_path

        print(f"Downloading {model_name} ({config['description']})...")
        print(f"  Size: ~{config['model_size_mb']} MB")

        # Download model file
        print(f"  Downloading {config['model_file']}...")
        self._download_file(config["model_url"], model_path)

        # Download vocab file
        print(f"  Downloading {config['vocab_file']}...")
        self._download_file(config["vocab_url"], vocab_path)

        print(f"✓ Model downloaded to {model_path}")

        # Save metadata
        self._save_metadata(model_dir, model_name, config)

        return model_path

    def _download_file(self, url: str, output_path: Path):
        """Download a file with progress indication.

        Args:
            url: URL to download from
            output_path: Local path to save file

        Raises:
            RuntimeError: If download fails
        """
        try:
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                if total_size > 0:
                    downloaded = block_num * block_size
                    percent = min(100, downloaded * 100 // total_size)
                    bars = "=" * (percent // 2)
                    print(f"\r    [{bars:<50}] {percent}%", end="", flush=True)

            urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
            print()  # New line after progress

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

    def _save_metadata(self, model_dir: Path, model_name: str, config: Dict):
        """Save model metadata.

        Args:
            model_dir: Model directory
            model_name: Model name
            config: Model configuration
        """
        metadata = {
            "model_name": model_name,
            "hf_repo": config["hf_repo"],
            "description": config["description"],
        }

        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_available_models(self) -> Dict[str, Dict]:
        """List available models.

        Returns:
            Dictionary of model names to configs
        """
        return MODEL_REGISTRY.copy()

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to cached model if it exists.

        Args:
            model_name: Name of model

        Returns:
            Path to model file, or None if not cached
        """
        if model_name not in MODEL_REGISTRY:
            return None

        config = MODEL_REGISTRY[model_name]
        model_path = self.cache_dir / model_name / config["model_file"]

        if model_path.exists():
            return model_path

        return None

    def verify_model(self, model_name: str) -> bool:
        """Verify that a cached model exists and is valid.

        Args:
            model_name: Name of model

        Returns:
            True if model is valid
        """
        if model_name not in MODEL_REGISTRY:
            return False

        config = MODEL_REGISTRY[model_name]
        model_dir = self.cache_dir / model_name

        model_path = model_dir / config["model_file"]
        vocab_path = model_dir / config["vocab_file"]

        # Check files exist
        if not model_path.exists() or not vocab_path.exists():
            return False

        # Check files are not empty
        if model_path.stat().st_size == 0 or vocab_path.stat().st_size == 0:
            return False

        return True

    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache.

        Args:
            model_name: Specific model to clear, or None for all models
        """
        if model_name is None:
            # Clear all models
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("All model caches cleared")
        else:
            # Clear specific model
            if model_name not in MODEL_REGISTRY:
                raise ValueError(f"Unknown model: {model_name}")

            model_dir = self.cache_dir / model_name
            if model_dir.exists():
                import shutil
                shutil.rmtree(model_dir)
                print(f"Model {model_name} cache cleared")


def download_model_cli():
    """CLI entry point for model downloader."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download embedding models for Prestige benchmarks"
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model name to download (default: bge-small)",
        default="bge-small",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Cache directory (default: ~/.cache/prestige/models)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify cached model",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear model cache",
    )

    args = parser.parse_args()

    downloader = ModelDownloader(args.cache_dir)

    if args.list:
        print("Available models:")
        for name, config in downloader.list_available_models().items():
            cached = "✓" if downloader.get_model_path(name) else " "
            print(f"  [{cached}] {name}: {config['description']} (~{config['model_size_mb']} MB)")
        return

    if args.clear:
        downloader.clear_cache(args.model if args.model != "bge-small" else None)
        return

    if args.verify:
        if downloader.verify_model(args.model):
            print(f"✓ Model {args.model} is valid")
        else:
            print(f"✗ Model {args.model} is invalid or not cached")
            exit(1)
        return

    # Download model
    try:
        model_path = downloader.download_model(args.model, force=args.force)
        print(f"\n✓ Success! Model path: {model_path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    download_model_cli()
