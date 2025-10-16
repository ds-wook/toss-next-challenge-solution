from itertools import combinations

import polars as pl


def build_skewed_features(df: pl.DataFrame, eps: float = 1e05) -> pl.DataFrame:
    """
    Build skewed features
    Args:
        df: polars DataFrame
        skewed_features: list of skewed features
        eps: epsilon
    Returns:
        polars DataFrame
    """
    b_features = ["feat_b_1", "feat_b_3", "feat_b_6"]

    df = df.with_columns(
        [
            (pl.col(f1) / (pl.col(f2) + eps)).alias(f"{f1}_div_{f2}")
            for f1, f2 in combinations(b_features, 2)
        ],
    )
    e_features = ["feat_e_2", "feat_e_7", "feat_e_10"]
    df = df.with_columns(
        [
            (pl.col(f1) / (pl.col(f2) + eps)).alias(f"{f1}_div_{f2}")
            for f1, f2 in combinations(e_features, 2)
        ]
    )

    return df
