import polars as pl


def denoise_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Denoise features with improved sparse handling
    Args:
        df: polars DataFrame
        steps: Dictionary mapping column names to denoising steps
    Returns:
        polars DataFrame
    """
    # Convert feat_a columns to string (for categorical encoding)
    denoise_features = [
        col
        for col in df.columns
        if col.startswith("feat_a")
        if col not in ["feat_a_5", "feat_a_14"]
    ]
    df = df.with_columns(
        [(pl.col(col).round().cast(pl.Int32)).alias(col) for col in denoise_features]
    )

    df = df.with_columns((pl.col("feat_e_4") * -100).cast(pl.Int32).alias("feat_e_4"))

    # Convert to appropriate data types
    df = df.with_columns([pl.col("hour").cast(pl.Int8)])
    df = df.with_columns((pl.col("hour") - 23).abs().alias("hour"))

    # feat_d_* 피처들을 label encoding으로 메모리 최적화
    # 1000배 스케일링 후 label encoding (고유값을 0부터 시작하는 정수로 매핑)
    df = df.with_columns(
        [
            (pl.col(col) * 1000).round(3).alias(col)
            for col in ["feat_d_1", "feat_d_2", "feat_d_3"]
        ]
    )
    df = df.with_columns(
        [
            (pl.col("feat_d_5") * -1000).round(3).alias("feat_d_5"),
            (pl.col("feat_d_6") * -1000).round(3).alias("feat_d_6"),
        ]
    )

    denoise_features = denoise_features + [
        "feat_d_1",
        "feat_d_2",
        "feat_d_3",
        "feat_d_5",
        "feat_d_6",
        "feat_e_4",
    ]

    # Label encoding: rank를 사용하여 고유값을 연속된 정수로 변환
    df = df.with_columns(
        [
            pl.col(col).rank(method="dense").cast(pl.Int16).alias(col)
            for col in denoise_features
        ]
    )

    return df


def denoise_history_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Denoise history features
    Args:
        df: polars DataFrame
    Returns:
        polars DataFrame
    """
    df = df.with_columns(
        [
            (pl.col("history_a_1") / (pl.col("history_a_2") + 1e05)).alias(
                "history_a1_a2_ratio"
            ),
            (pl.col("history_a_3") / (pl.col("history_a_4") + 1e05)).alias(
                "history_a3_a4_ratio"
            ),
        ]
    )
    history_b_cols = [col for col in df.columns if col.startswith("history_b")]

    # Recent K개 집계 (예: last 3)
    df = df.with_columns(
        [
            pl.sum_horizontal(*history_b_cols[-3:]).alias("history_b_last3_sum"),
            pl.mean_horizontal(*history_b_cols[-3:]).alias("history_b_last3_mean"),
            pl.max_horizontal(*history_b_cols[-3:]).alias("history_b_last3_max"),
        ]
    )

    # 5Ratio (최근 2개 비교)
    df = df.with_columns(
        [
            (pl.col(history_b_cols[-1]) / (pl.col(history_b_cols[-2]) + 1e05)).alias(
                "history_b_last_ratio"
            )
        ]
    )

    df = df.with_columns(
        [
            pl.when(pl.col(c).is_infinite())  # +inf, -inf 체크
            .then(None)  # NaN/Null 로 치환
            .otherwise(pl.col(c))
            .alias(c)
            for c in [
                "history_b_last3_sum",
                "history_b_last3_mean",
                "history_b_last3_max",
                "history_b_last_ratio",
            ]
        ]
    )

    return df
