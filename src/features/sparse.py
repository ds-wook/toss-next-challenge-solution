import polars as pl


def add_sparsity_features(df: pl.DataFrame, sparse_cols: list[str]) -> pl.DataFrame:
    """
    Add sparsity-related features for columns with many zeros

    Args:
        df: polars DataFrame
        sparse_cols: List of column names that are expected to be sparse

    Returns:
        polars DataFrame with additional sparsity features
    """
    # Create expressions for sparsity features
    sparsity_expressions = []

    for col in sparse_cols:
        if col in df.columns:
            # Non-zero indicator (1 if non-zero, 0 if zero)
            sparsity_expressions.append(
                (pl.col(col) != 0.0).cast(pl.Int8).alias(f"{col}_non_zero")
            )

            # Zero indicator (1 if zero, 0 if non-zero)
            sparsity_expressions.append(
                (pl.col(col) == 0.0).cast(pl.Int8).alias(f"{col}_zero")
            )

            # Magnitude indicator (1 if value > 0, -1 if value < 0, 0 if value == 0)
            sparsity_expressions.append(
                pl.when(pl.col(col) > 0)
                .then(1)
                .when(pl.col(col) < 0)
                .then(-1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias(f"{col}_magnitude")
            )

    # Add all sparsity features to dataframe at once
    if sparsity_expressions:
        df = df.with_columns(sparsity_expressions)

    return df


def create_sparse_interactions(
    df: pl.DataFrame, sparse_cols: list[str]
) -> pl.DataFrame:
    """
    Create interaction features between sparse columns

    Args:
        df: polars DataFrame
        sparse_cols: List of sparse column names

    Returns:
        polars DataFrame with interaction features
    """
    # Count of non-zero sparse features per row
    non_zero_cols = [
        f"{col}_non_zero" for col in sparse_cols if f"{col}_non_zero" in df.columns
    ]

    if non_zero_cols:
        df = df.with_columns(
            pl.sum_horizontal(non_zero_cols).alias("sparse_non_zero_count")
        )

        # Ratio of non-zero sparse features
        df = df.with_columns(
            (pl.col("sparse_non_zero_count") / len(non_zero_cols)).alias(
                "sparse_non_zero_ratio"
            )
        )

    # Sum of non-zero sparse values
    sparse_sum_cols = [col for col in sparse_cols if col in df.columns]
    if sparse_sum_cols:
        df = df.with_columns(pl.sum_horizontal(sparse_sum_cols).alias("sparse_sum"))

        # Mean of non-zero sparse values (excluding zeros)
        df = df.with_columns(
            pl.when(pl.col("sparse_non_zero_count") > 0)
            .then(pl.col("sparse_sum") / pl.col("sparse_non_zero_count"))
            .otherwise(0.0)
            .alias("sparse_mean_non_zero")
        )

    return df


def optimize_sparse_encoding(df: pl.DataFrame, sparse_cols: list[str]) -> pl.DataFrame:
    """
    Optimize encoding for sparse columns by combining similar approaches

    Args:
        df: polars DataFrame
        sparse_cols: List of sparse column names

    Returns:
        polars DataFrame with optimized sparse encoding
    """
    # Step 1: Add sparsity features
    df = add_sparsity_features(df, sparse_cols)

    # Step 2: Create interactions
    df = create_sparse_interactions(df, sparse_cols)

    return df
