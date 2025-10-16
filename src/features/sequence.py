from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl
from tqdm import tqdm


def build_seq_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Build sequence statistics with optional advanced features"""

    def extract_seq_stats(seq_str: str) -> dict[str, float | str]:
        """Extract sequence statistics"""
        items = seq_str.split(",")
        seq_unique = len(set(items))
        diversity_ratio = len(set(items)) / len(items)

        # 엔트로피 계산
        if len(items) == 0:
            entropy = 0.0
        else:
            item_counts = Counter(items)
            total_count = len(items)
            entropy = -np.sum(
                (count / total_count) * np.log2(count / total_count)
                for count in item_counts.values()
            )

        stats = {
            "seq_len": len(items),
            "seq_unique": seq_unique,
            "diversity_ratio": diversity_ratio,
            "entropy": entropy,
            "seq_last": items[-1],
        }

        return stats

    # 진행률 표시를 위한 wrapper 함수
    def extract_seq_stats_with_progress(seq_str: str) -> dict[str, float | str]:
        """Extract sequence statistics with progress bar"""
        return extract_seq_stats(seq_str)

    # 데이터 전처리 - return_dtype을 명시적으로 지정
    # tqdm을 사용해서 진행률 표시
    seq_data = df["seq"].to_list()
    processed_data = []

    print("Processing sequence statistics...")
    for seq_str in tqdm(seq_data, desc="Building seq stats"):
        processed_data.append(extract_seq_stats(seq_str))

    # 결과를 DataFrame으로 변환
    result_df = pl.DataFrame(processed_data)

    # 원본 DataFrame과 결합
    df = df.with_columns(
        [
            result_df["seq_len"].cast(pl.Int64).alias("seq_len"),
            result_df["seq_unique"].cast(pl.Int64).alias("seq_unique"),
            result_df["diversity_ratio"].cast(pl.Float64).alias("diversity_ratio"),
            result_df["entropy"].cast(pl.Float64).alias("entropy"),
            result_df["seq_last"].cast(pl.Int64).alias("seq_last"),
        ]
    )

    return df
