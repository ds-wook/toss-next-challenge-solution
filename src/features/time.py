from __future__ import annotations

import numpy as np
import polars as pl


def build_time_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Build time-related features from day_of_week and hour columns

    Args:
        df: polars DataFrame with 'day_of_week' and 'hour' columns

    Returns:
        polars DataFrame with additional time features
    """

    # Convert to appropriate data types
    df = df.with_columns([pl.col("hour").cast(pl.Int8)])
    df = df.with_columns((pl.col("hour") - 23).abs().alias("hour"))
    # Basic time features
    df = df.with_columns(
        [
            # Time period features
            pl.when(pl.col("hour").is_between(6, 11))
            .then(1)
            .when(pl.col("hour").is_between(12, 17))
            .then(2)
            .when(pl.col("hour").is_between(18, 23))
            .then(3)
            .otherwise(0)
            .alias("time_period"),  # 0: night, 1: morning, 2: afternoon, 3: evening
            # Business hours
            pl.when(pl.col("hour").is_between(9, 17))
            .then(1)
            .otherwise(0)
            .alias("is_business_hours"),
            # Peak hours (assuming 12-14 and 18-20 are peak)
            pl.when(pl.col("hour").is_in([12, 13, 14, 18, 19, 20]))
            .then(1)
            .otherwise(0)
            .alias("is_peak_hours"),
        ]
    )
    # Cyclical encoding for time features
    df = df.with_columns(
        [
            # Cyclical encoding for hour (0-23 -> 0-2π)
            (2 * np.pi * pl.col("hour") / 24).alias("hour_cyclical"),
            (2 * np.pi * pl.col("hour") / 24).sin().alias("hour_sin"),
            (2 * np.pi * pl.col("hour") / 24).cos().alias("hour_cos"),
            # Half-day and quarter-day cycles
            (2 * np.pi * pl.col("hour") / 12).sin().alias("half_day_sin"),
            (2 * np.pi * pl.col("hour") / 12).cos().alias("half_day_cos"),
            (2 * np.pi * pl.col("hour") / 6).sin().alias("quarter_day_sin"),
            (2 * np.pi * pl.col("hour") / 6).cos().alias("quarter_day_cos"),
        ]
    )

    return df
