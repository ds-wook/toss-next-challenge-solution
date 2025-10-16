"""
Data loading utilities for CTR prediction
"""

from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(
    data_path: str,
    use_sampling: bool = False,
    sampling_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load train and test data from parquet files with optional negative sampling

    Args:
        data_path: Path to data directory
        use_sampling: Whether to apply negative sampling
        sampling_ratio: Ratio of negative samples to keep (0.0-1.0)
        seed: Random seed for sampling

    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_parquet(f"{data_path}/train.parquet", engine="pyarrow")
    test_df = pd.read_parquet(f"{data_path}/test.parquet", engine="pyarrow")

    print(f"Original train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(
        f"Original target distribution: {train_df['clicked'].value_counts(normalize=True)}"
    )

    # Apply negative sampling if requested
    if use_sampling:
        train_df = apply_negative_sampling(train_df, sampling_ratio, seed)
        print(f"After sampling train shape: {train_df.shape}")
        print(
            f"After sampling target distribution: {train_df['clicked'].value_counts(normalize=True)}"
        )

    return train_df, test_df


def apply_negative_sampling(
    train_df: pd.DataFrame, sampling_ratio: float = 0.5, seed: int = 42
) -> pd.DataFrame:
    """
    Apply negative sampling to reduce the number of negative samples

    Args:
        train_df: Training dataframe
        sampling_ratio: Ratio of negative samples to keep (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Sampled dataframe with reduced negative samples
    """
    # Set random seed
    np.random.seed(seed)

    # Separate positive and negative samples
    positive_samples = train_df[train_df["clicked"] == 1]
    negative_samples = train_df[train_df["clicked"] == 0]

    print(
        f"Original - Positive: {len(positive_samples)}, Negative: {len(negative_samples)}"
    )

    # Sample negative samples
    if sampling_ratio < 1.0:
        n_negative_samples = int(len(negative_samples) * sampling_ratio)
        negative_samples_sampled = negative_samples.sample(
            n=n_negative_samples, random_state=seed
        )
    else:
        negative_samples_sampled = negative_samples

    # Combine positive and sampled negative samples
    sampled_df = pd.concat([positive_samples, negative_samples_sampled], axis=0)

    # Shuffle the combined dataset
    sampled_df = sampled_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(
        f"After sampling - Positive: {len(positive_samples)}, Negative: {len(negative_samples_sampled)}"
    )
    print(f"Sampling ratio applied: {sampling_ratio:.2f}")

    return sampled_df


def downsample_data(
    train_df: pd.DataFrame, ratio: float = 2.0, random_state: int = 42
) -> pd.DataFrame:
    """
    Downsample the majority class to balance the dataset

    Args:
        train_df: Training dataframe
        ratio: Ratio of negative to positive samples
        random_state: Random seed

    Returns:
        Downsampled dataframe
    """
    clicked_1 = train_df[train_df["clicked"] == 1]
    clicked_0 = train_df[train_df["clicked"] == 0]

    # Sample negative class
    sample_size = int(min(len(clicked_1) * ratio, len(clicked_0)))
    clicked_0_sampled = clicked_0.sample(n=sample_size, random_state=random_state)

    # Combine and shuffle
    balanced_df = pd.concat([clicked_1, clicked_0_sampled], axis=0)
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    print("After downsampling:")
    print(f"Train shape: {balanced_df.shape}")
    print(f"Clicked=0: {len(balanced_df[balanced_df['clicked'] == 0])}")
    print(f"Clicked=1: {len(balanced_df[balanced_df['clicked'] == 1])}")
    print(
        f"Ratio: {len(balanced_df[balanced_df['clicked'] == 0]) / len(balanced_df[balanced_df['clicked'] == 1]):.2f}:1"
    )

    return balanced_df


def get_feature_columns(
    train_df: pd.DataFrame, exclude_cols: List[str] = None
) -> List[str]:
    """
    Get feature columns excluding specified columns

    Args:
        train_df: Training dataframe
        exclude_cols: Columns to exclude from features

    Returns:
        List of feature column names
    """
    if exclude_cols is None:
        exclude_cols = ["clicked", "seq", "ID"]

    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    print(f"Number of features: {len(feature_cols)}")
    print(f"First 10 features: {feature_cols[:10]}")

    return feature_cols


def split_data(
    train_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split data into train and validation sets

    Args:
        train_df: Training dataframe
        target_col: Target column name
        test_size: Validation set size
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df)
    """
    train_split, val_split = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df[target_col],
        shuffle=True,
    )

    print(f"Train split: {train_split.shape}")
    print(f"Val split: {val_split.shape}")

    return train_split, val_split
