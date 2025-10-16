from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from data.base import BaseDataLoader
from features.build import build_skewed_features
from features.denoise import denoise_features, denoise_history_features


class TreeDataLoader(BaseDataLoader):
    """LightGBM 모델을 위한 데이터 로더"""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.count_mapping = {}  # train의 count를 저장

    def preprocess_train_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Train 데이터 전처리 - target encoding 포함"""
        df = self.preprocess_data(df, is_train=True)
        return df

    def preprocess_test_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Test 데이터 전처리 - 저장된 mapping으로 target encoding 적용"""
        df = self.preprocess_data(df, is_train=False)
        return df

    def preprocess_data(self, df: pl.DataFrame, is_train: bool = True) -> pl.DataFrame:
        """데이터 전처리 수행"""
        processing_steps = [
            ("Processing denoise features", denoise_features),
            ("Processing history features", denoise_history_features),
            ("Processing skewed features", build_skewed_features),
            ("Processing round trick", self._round_trick),
            ("Processing variables", self._encode_categorical_variables),
            (
                "Processing count encoding",
                lambda x: self._encode_categorical_count(x, is_train),
            ),
        ]

        with tqdm(total=len(processing_steps), desc="Processing data") as pbar:
            for description, step_func in processing_steps:
                pbar.set_description(description)
                df = step_func(df)
                pbar.update(1)

        return df

    def _encode_categorical_count(
        self, df: pl.DataFrame, is_train: bool = True
    ) -> pl.DataFrame:
        """범주형 변수에 대한 count encoding 수행

        Train 데이터의 count를 기준으로 test 데이터에도 동일하게 적용합니다.

        Args:
            df: 입력 데이터프레임
            is_train: Train 데이터 여부. True면 count를 계산하고 저장,
                     False면 저장된 count를 사용

        Returns:
            count encoding이 추가된 데이터프레임

        Example:
            >>> df = pl.DataFrame({
            ...     "gender": ["1.0", "2.0", "1.0", "1.0", "2.0"],
            ...     "age": [20, 30, 20, 20, 30]
            ... })
            >>> loader._encode_categorical_count(df, is_train=True)
            # gender_count: [3, 2, 3, 3, 2]
            # age_count: [3, 2, 3, 3, 2]
        """
        # 각 범주형 변수에 대해 count encoding 적용
        for col in self.cfg.store.cat_features:
            if is_train:
                # Train: count 계산 후 저장
                count_map = (
                    df.group_by(col)
                    .agg(pl.count().alias("count"))
                    .select([pl.col(col), pl.col("count")])
                )
                self.count_mapping[col] = count_map

            else:
                # Test: 저장된 count 사용
                if col in self.count_mapping:
                    count_map = self.count_mapping[col]
                    df = df.join(
                        count_map.rename({"count": f"{col}_count"}), on=col, how="left"
                    )
                    # Train에 없던 값은 0으로 처리
                    df = df.with_columns(pl.col(f"{col}_count").fill_null(0))

                else:
                    # 저장된 count가 없으면 0으로 초기화
                    df = df.with_columns(pl.lit(0).alias(f"{col}_count"))

        return df

    def _encode_categorical_variables(self, df: pl.DataFrame) -> pl.DataFrame:
        """범주형 변수를 정수형으로 변환 - 실제 값의 범위에 따라 타입 자동 선택"""
        # (min, max, dtype) 매핑: -1은 null 값으로 예약
        dtype_ranges = [
            (-128, 126, pl.Int8),  # Int8: -128 ~ 127, -1 예약
            (-32768, 32766, pl.Int16),  # Int16: -32,768 ~ 32,767, -1 예약
            (float("-inf"), float("inf"), pl.Int32),  # Int32: 나머지
        ]

        for col in self.cfg.store.cat_features:
            col_dtype = df[col].dtype

            # 문자열인 경우: 먼저 Int32로 변환 (Int8은 문자열 변환 실패)
            if not col_dtype.is_numeric():
                df = df.with_columns(
                    pl.col(col).str.replace(".0", "").cast(pl.Int32).alias(col)
                )

            # null을 -1로 먼저 처리
            df = df.with_columns(pl.col(col).fill_null(-1).alias(col))

            # 실제 min/max 값 확인
            col_min = df[col].min()
            col_max = df[col].max()

            # 값의 범위에 맞는 최적 dtype 선택
            dtype = next(
                dtype
                for min_val, max_val, dtype in dtype_ranges
                if col_min >= min_val and col_max <= max_val
            )

            # 최적 dtype으로 캐스팅
            df = df.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))

        return df

    def save_count_mapping(self, save_path: str | Path) -> None:
        """count_mapping을 pickle 파일로 저장

        Args:
            save_path: 저장할 파일 경로 (예: "res/data/count_mapping.pkl")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self.count_mapping, f)
        print(f"Count mapping saved to {save_path}")

    def load_count_mapping(self, load_path: str | Path) -> None:
        """pickle 파일에서 count_mapping 로드

        Args:
            load_path: 로드할 파일 경로 (예: "res/data/count_mapping.pkl")
        """
        load_path = Path(load_path)

        if not load_path.exists():
            print(f"Warning: Count mapping file not found at {load_path}")
            self.count_mapping = {}
            return

        with open(load_path, "rb") as f:
            self.count_mapping = pickle.load(f)
        print(f"Count mapping loaded from {load_path}")
