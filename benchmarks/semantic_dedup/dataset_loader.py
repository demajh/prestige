"""Dataset loading and preprocessing for semantic deduplication benchmarks."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Iterator, Optional, Any
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for a benchmark dataset."""
    name: str
    source: str
    text_columns: List[str]
    label_column: str
    positive_label: Any
    sample_size: Optional[int] = None
    sts_threshold: float = 4.0  # For STS-B: similarity >= 4.0 means duplicate


# Pre-configured benchmark datasets
DATASETS = {
    "qqp": DatasetConfig(
        name="qqp",
        source="glue",
        text_columns=["question1", "question2"],
        label_column="label",
        positive_label=1,
        sample_size=None,  # ~400k pairs
    ),
    "mrpc": DatasetConfig(
        name="mrpc",
        source="glue",
        text_columns=["sentence1", "sentence2"],
        label_column="label",
        positive_label=1,
        sample_size=None,  # ~5.8k pairs - good for quick iteration
    ),
    "stsb": DatasetConfig(
        name="stsb",
        source="glue",
        text_columns=["sentence1", "sentence2"],
        label_column="label",
        positive_label=None,  # Graded 0-5, will threshold at 4.0
        sample_size=None,
        sts_threshold=4.0,
    ),
    "paws": DatasetConfig(
        name="paws",
        source="paws",
        text_columns=["sentence1", "sentence2"],
        label_column="label",
        positive_label=1,
        sample_size=None,  # ~108k adversarial pairs
    ),
    "paranmt": DatasetConfig(
        name="paranmt",
        source="para_nmt",  # May need alternative source
        text_columns=["text1", "text2"],
        label_column="label",
        positive_label=1,
        sample_size=100000,  # Sample 100k from millions
    ),
}


class DatasetLoader:
    """Handles dataset download, caching, and preprocessing."""

    def __init__(self, cache_dir: Path):
        """Initialize dataset loader.

        Args:
            cache_dir: Directory for caching downloaded datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / "manifest.json"

    def load_dataset(self, config: DatasetConfig) -> pd.DataFrame:
        """Load dataset from cache or download if needed.

        Args:
            config: Dataset configuration

        Returns:
            DataFrame with columns: text1, text2, is_duplicate
        """
        # Check cache
        cache_file = self.cache_dir / f"{config.name}.parquet"
        if cache_file.exists() and self._is_cache_valid(config):
            print(f"Loading {config.name} from cache...")
            return pd.read_parquet(cache_file)

        # Download and process
        print(f"Downloading {config.name} dataset...")
        df = self._download_dataset(config)

        # Save to cache
        df.to_parquet(cache_file)
        self._update_manifest(config)

        return df

    def _download_dataset(self, config: DatasetConfig) -> pd.DataFrame:
        """Download dataset from HuggingFace.

        Args:
            config: Dataset configuration

        Returns:
            DataFrame with normalized columns
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library required. Install with: pip install datasets"
            )

        # Load from HuggingFace
        if config.source == "glue":
            dataset = load_dataset("glue", config.name, split="train")
        elif config.source == "paws":
            dataset = load_dataset("paws", "labeled_final", split="train")
        elif config.source == "para_nmt":
            # ParaNMT may require special handling or alternative source
            raise NotImplementedError(
                "ParaNMT dataset requires custom download logic. "
                "Consider using an alternative large-scale paraphrase dataset."
            )
        else:
            dataset = load_dataset(config.source, split="train")

        # Convert to pandas
        df = dataset.to_pandas()

        # Normalize columns
        df = self._normalize_columns(df, config)

        # Apply sampling if needed
        if config.sample_size and len(df) > config.sample_size:
            df = self._stratified_sample(df, config.sample_size)

        return df

    def _normalize_columns(self, df: pd.DataFrame, config: DatasetConfig) -> pd.DataFrame:
        """Normalize dataset columns to standard format.

        Args:
            df: Input dataframe
            config: Dataset configuration

        Returns:
            DataFrame with columns: text1, text2, is_duplicate
        """
        # Rename text columns
        result = pd.DataFrame()
        result["text1"] = df[config.text_columns[0]]
        result["text2"] = df[config.text_columns[1]]

        # Process labels
        if config.name == "stsb":
            # STS-B has graded similarity scores (0-5)
            # Threshold at 4.0 to binarize
            result["is_duplicate"] = df[config.label_column] >= config.sts_threshold
        else:
            # Binary labels
            result["is_duplicate"] = df[config.label_column] == config.positive_label

        # Remove rows with null values
        result = result.dropna()

        return result

    def _stratified_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """Perform stratified sampling to preserve class balance.

        Args:
            df: Input dataframe
            sample_size: Number of samples to draw

        Returns:
            Sampled dataframe
        """
        # Group by label
        positive = df[df["is_duplicate"] == True]
        negative = df[df["is_duplicate"] == False]

        # Calculate sample sizes
        pos_ratio = len(positive) / len(df)
        pos_sample_size = int(sample_size * pos_ratio)
        neg_sample_size = sample_size - pos_sample_size

        # Sample from each group
        pos_sample = positive.sample(n=min(pos_sample_size, len(positive)), random_state=42)
        neg_sample = negative.sample(n=min(neg_sample_size, len(negative)), random_state=42)

        # Combine and shuffle
        result = pd.concat([pos_sample, neg_sample])
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)

        return result

    def _is_cache_valid(self, config: DatasetConfig) -> bool:
        """Check if cached dataset is still valid.

        Args:
            config: Dataset configuration

        Returns:
            True if cache is valid
        """
        if not self.manifest_path.exists():
            return False

        with open(self.manifest_path, "r") as f:
            manifest = json.load(f)

        if config.name not in manifest:
            return False

        # Check if config matches
        cached_config = manifest[config.name]
        current_config = asdict(config)

        return cached_config == current_config

    def _update_manifest(self, config: DatasetConfig):
        """Update manifest with dataset info.

        Args:
            config: Dataset configuration
        """
        manifest = {}
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                manifest = json.load(f)

        manifest[config.name] = asdict(config)

        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def get_text_pairs(
        self, df: pd.DataFrame
    ) -> Iterator[Tuple[str, str, bool]]:
        """Iterate over text pairs from dataset.

        Args:
            df: DataFrame with text1, text2, is_duplicate columns

        Yields:
            Tuples of (text1, text2, is_duplicate)
        """
        for _, row in df.iterrows():
            yield (
                str(row["text1"]),
                str(row["text2"]),
                bool(row["is_duplicate"])
            )


def list_available_datasets() -> List[str]:
    """Get list of available dataset names.

    Returns:
        List of dataset names
    """
    return list(DATASETS.keys())


def get_dataset_config(name: str) -> DatasetConfig:
    """Get configuration for a dataset.

    Args:
        name: Dataset name

    Returns:
        Dataset configuration

    Raises:
        ValueError: If dataset name not found
    """
    if name not in DATASETS:
        available = ", ".join(list_available_datasets())
        raise ValueError(
            f"Unknown dataset: {name}. Available datasets: {available}"
        )
    return DATASETS[name]
