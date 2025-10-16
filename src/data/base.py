from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import polars as pl
import torch
from easydict import EasyDict
from omegaconf import DictConfig
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch import Tensor
from tqdm import tqdm

from models.cv import LeaveOneDayOutCV
from utils.utils import reduce_mem_usage


@dataclass
class DataConfig:
    """데이터 설정을 위한 데이터 클래스"""

    target_col: str
    data_path: str
    sampling_ratio: float = 1.0
    seed: int = 42
    n_splits: int = 5
    train_test_split: bool = True
    test_size: float = 0.2
    stratify: bool = True
    split_type: str = "stratified"
    group: str = "day_of_week"
    fold_idx_for_fm: int = 0


class BaseDataLoader(ABC):
    """데이터 로딩을 위한 기본 클래스"""

    def __init__(self, cfg: DictConfig | EasyDict):
        self.cfg = cfg
        self.data_config = DataConfig(
            target_col=cfg.data.target,
            data_path=cfg.data.path,
            sampling_ratio=cfg.data.sampling,
            seed=cfg.data.seed,
            n_splits=cfg.data.n_splits,
            train_test_split=getattr(cfg.data, "train_test_split", True),
            test_size=getattr(cfg.data, "test_size", 0.2),
            stratify=getattr(cfg.data, "stratify", True),
            split_type=getattr(cfg.data, "split_type", "stratified"),
            group=getattr(cfg.data, "group", "day_of_week"),
            fold_idx_for_fm=getattr(cfg.data, "fold_idx_for_fm", 0),
        )
        self.train_data: pl.DataFrame | None = None
        self.test_data: pl.DataFrame | None = None
        self.feature_columns: list[str] | None = None
        self.target_column: str | None = None

    def load_train_data(
        self,
        is_boosting: bool = False,
        is_fm: bool = False,
        is_test: bool = False,
    ) -> tuple[pl.DataFrame, pl.Series] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """훈련 데이터 로드"""
        print("훈련 데이터 로딩 중...")

        # 훈련 데이터 로드
        train_path = Path(self.cfg.data.path) / f"{self.cfg.data.train}.parquet"
        print(f"Loading training data from: {train_path}")
        train = pl.read_parquet(train_path)

        # randomly select 10000 rows for quick test
        if is_test:
            train = train.sample(n=10000, seed=self.data_config.seed)

        # 메모리 최적화 적용
        train = reduce_mem_usage(train, verbose=True)

        if is_boosting:
            # preprocess train data
            train = self.preprocess_train_data(train)

            train_x = train.drop([self.cfg.data.target])
            train_y = train[self.cfg.data.target]

            return train_x, train_y

        if is_fm:
            train = train.select(
                self.cfg.data.num_features
                + self.cfg.data.cat_features
                + [self.cfg.data.target, self.cfg.data.seq]
            )

            self._get_min_max_id_in_seq_feature(train)

            train_indices, val_indices = self._kfold_split_indices(
                X=train.select(
                    ["l_feat_1"]
                ).to_pandas(),  # arbitrary column for splitting
                y=train.select(self.cfg.data.target).to_pandas(),
                groups=train[self.cfg.data.group].to_pandas()
                if self.cfg.data.group not in self.cfg.data.drop_features
                else None,
            )
            train, val = train[train_indices], train[val_indices]

            train_seq = train[self.cfg.data.seq]
            val_seq = val[self.cfg.data.seq]

            train = train.drop(self.cfg.data.seq)
            val = val.drop(self.cfg.data.seq)

            # preprocess train data
            train = self.preprocess_train_data(train)
            # preprocess val data using statistics or scaler from train data
            val = self.preprocess_test_data(val, is_validation=True)

            train_x = train.select(
                self.cfg.data.num_features + self.cfg.data.cat_features
            )
            train_y = train.select(self.cfg.data.target)
            val_x = val.select(self.cfg.data.num_features + self.cfg.data.cat_features)
            val_y = val.select(self.cfg.data.target)

            # convert to tensor for future training
            train_x = torch.tensor(train_x.to_numpy(), dtype=torch.float32)
            train_y = torch.tensor(train_y.to_numpy(), dtype=torch.float32)
            val_x = torch.tensor(val_x.to_numpy(), dtype=torch.float32)
            val_y = torch.tensor(val_y.to_numpy(), dtype=torch.float32)

            if self.cfg.data.use_seq_feature:
                return (
                    train_x,
                    train_y,
                    train_seq.to_list(),
                    val_x,
                    val_y,
                    val_seq.to_list(),
                )
            else:
                return (
                    train_x,
                    train_y,
                    None,  # seq loc
                    val_x,
                    val_y,
                    None,  # seq loc
                )

    def load_test_data(
        self,
        is_boosting: bool = False,
        is_fm: bool = False,
    ) -> pl.DataFrame | tuple[Tensor, list]:
        """테스트 데이터 로드"""
        print("테스트 데이터 로딩 중...")

        # 테스트 데이터 로드 (별도 파일이 있는 경우)
        test_path = Path(self.cfg.data.path) / f"{self.cfg.data.test}.parquet"
        test = pl.read_parquet(test_path)

        test_seq = test[self.cfg.data.seq]

        # 메모리 최적화 적용
        test = reduce_mem_usage(test, verbose=True)

        # preprocess test data
        test = self.preprocess_test_data(test)

        if is_boosting:
            test_x = test.drop([self.cfg.data.id, *self.cfg.data.drop_features])

            return test_x

        if is_fm:
            test_x = test.select(
                self.cfg.data.num_features + self.cfg.data.cat_features
            )
            test_x = torch.tensor(test_x.to_numpy(), dtype=torch.float32)

            if self.cfg.data.use_seq_feature:
                return (
                    test_x,
                    test["ID"].to_list(),
                    test_seq.to_list(),
                )
            else:
                return test_x, test["ID"].to_list(), None

    @abstractmethod
    def preprocess_train_data(self) -> None:
        """데이터 전처리를 수행하는 메서드"""
        raise NotImplementedError

    @abstractmethod
    def preprocess_test_data(self) -> None:
        """데이터 전처리를 수행하는 메서드"""
        raise NotImplementedError

    def split_train_test(self, data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """훈련 데이터를 train/test로 분리"""
        if not self.data_config.train_test_split:
            return data, pl.DataFrame()

        # 타겟 컬럼이 있는지 확인
        if self.data_config.target_col not in data.columns:
            raise ValueError(
                f"타겟 컬럼 '{self.data_config.target_col}'이 데이터에 없습니다."
            )

        # Stratified split을 위한 타겟 분포 확인
        if self.data_config.stratify:
            target_counts = data[self.data_config.target_col].value_counts()
            if len(target_counts) < 2:
                print(
                    "경고: 타겟이 하나의 클래스만 있어서 stratify를 사용할 수 없습니다. 랜덤 분할을 사용합니다."
                )
                self.data_config.stratify = False

        # Polars에서 train/test 분리
        if self.data_config.stratify:
            # Stratified split 구현
            train_data, test_data = self._stratified_split(data)
        else:
            # 랜덤 분할
            train_data, test_data = self._random_split(data)

        print("Train/Test 분리 완료:")
        print(f"  Train: {train_data.shape}")
        print(f"  Test: {test_data.shape}")

        return train_data, test_data

    def _stratified_split(
        self, data: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Stratified train/test 분리"""
        # 각 클래스별로 분리
        target_col = self.data_config.target_col
        unique_targets = data[target_col].unique().sort()

        train_dfs = []
        test_dfs = []

        for target_val in unique_targets:
            class_data = data.filter(pl.col(target_col) == target_val)
            class_size = len(class_data)
            test_size = int(class_size * self.data_config.test_size)

            # 클래스별로 랜덤 샘플링
            shuffled = class_data.sample(fraction=1.0, seed=self.data_config.seed)
            test_class = shuffled.head(test_size)
            train_class = shuffled.tail(class_size - test_size)

            train_dfs.append(train_class)
            test_dfs.append(test_class)

        # 결과 합치기
        train_data = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
        test_data = pl.concat(test_dfs) if test_dfs else pl.DataFrame()

        return train_data, test_data

    def _stratified_split_indices(
        self, data: pl.DataFrame
    ) -> tuple[list[int], list[int]]:
        """Stratified train/test 분리 - 인덱스 반환"""
        # 각 클래스별로 분리
        target_col = self.data_config.target_col
        unique_targets = data[target_col].unique().sort()

        train_indices = []
        test_indices = []

        # 원본 데이터에 행 인덱스 추가
        data_with_idx = data.with_row_index("__row_idx__")

        for target_val in unique_targets:
            class_data = data_with_idx.filter(pl.col(target_col) == target_val)
            class_size = len(class_data)
            test_size = int(class_size * self.data_config.test_size)

            # 클래스별로 랜덤 샘플링
            shuffled = class_data.sample(fraction=1.0, seed=self.data_config.seed)
            test_class_indices = shuffled.head(test_size)["__row_idx__"].to_list()
            train_class_indices = shuffled.tail(class_size - test_size)[
                "__row_idx__"
            ].to_list()

            train_indices.extend(train_class_indices)
            test_indices.extend(test_class_indices)

        return train_indices, test_indices

    def _kfold_split_indices(
        self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
    ) -> list[tuple[list[int], list[int]]]:
        match self.data_config.split_type:
            case "day_of_week":
                kfold = LeaveOneDayOutCV(day_col="day_of_week")
                k_splits = kfold.split(X)
            case "group":
                kfold = StratifiedGroupKFold(
                    n_splits=self.n_splits,
                    shuffle=True,
                    random_state=self.data_config.seed,
                )
                k_splits = kfold.split(X, y, groups=groups)
            case _:
                kfold = StratifiedKFold(
                    n_splits=self.data_config.n_splits,
                    shuffle=True,
                    random_state=self.data_config.seed,
                )
                k_splits = kfold.split(X, y)

        with tqdm(k_splits, total=kfold.get_n_splits(X, y)) as pbar:
            for fold, (train_idx, valid_idx) in enumerate(pbar, 1):
                if fold == self.data_config.fold_idx_for_fm:
                    print(f"Using fold {fold} for FM training")
                    return train_idx, valid_idx
        raise ValueError("Fold index for FM training is out of range.")

    def _random_split(self, data: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """랜덤 train/test 분리"""
        total_size = len(data)
        test_size = int(total_size * self.data_config.test_size)

        # 전체 데이터를 랜덤하게 섞기
        shuffled = data.sample(fraction=1.0, seed=self.data_config.seed)

        # 분리
        test_data = shuffled.head(test_size)
        train_data = shuffled.tail(total_size - test_size)

        return train_data, test_data

    def get_feature_columns(self) -> list[str]:
        """피처 컬럼 목록을 반환"""
        if self.feature_columns is None:
            raise ValueError(
                "데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요."
            )
        return self.feature_columns

    def get_target_column(self) -> str:
        """타겟 컬럼명을 반환"""
        if self.target_column is None:
            raise ValueError(
                "데이터가 로드되지 않았습니다. load_data()를 먼저 호출하세요."
            )
        return self.target_column

    def negative_sampling_train_dataset(self, df: pl.DataFrame) -> pl.DataFrame:
        # Read all data at once with Polars (more efficient than chunking)
        with tqdm(total=4, desc="Processing dataset", unit="step") as pbar:
            # Step 1: Load data
            pbar.set_description("Reading parquet file")
            start_time = time.time()
            load_time = time.time() - start_time
            pbar.set_postfix({"rows": f"{len(df):,}", "time": f"{load_time:.2f}s"})
            pbar.update(1)

            # Step 2: Filter data
            pbar.set_description("Filtering samples")
            start_time = time.time()
            positive = df.filter(pl.col(self.cfg.data.target) == 1)
            negative = df.filter(pl.col(self.cfg.data.target) == 0)
            filter_time = time.time() - start_time
            pbar.set_postfix(
                {
                    "positive": f"{len(positive):,}",
                    "negative": f"{len(negative):,}",
                    "time": f"{filter_time:.2f}s",
                }
            )
            pbar.update(1)

            # Step 3: Sample negative data
            pbar.set_description(
                f"Sampling {self.cfg.data.sampling:.1%} of negative data"
            )
            start_time = time.time()
            negative_sample = negative.sample(
                fraction=self.cfg.data.sampling,
                with_replacement=False,
                seed=self.cfg.data.seed,
            )
            sample_time = time.time() - start_time
            pbar.set_postfix(
                {"sampled": f"{len(negative_sample):,}", "time": f"{sample_time:.2f}s"}
            )
            pbar.update(1)

            # Step 4: Combine datasets
            pbar.set_description("Combining datasets")
            start_time = time.time()
            train = pl.concat([negative_sample, positive])
            combine_time = time.time() - start_time
            pbar.set_postfix(
                {
                    "total": f"{len(train):,}",
                    "positive": f"{len(positive):,}",
                    "negative": f"{len(negative_sample):,}",
                    "time": f"{combine_time:.2f}s",
                }
            )
            pbar.update(1)

        return train

    def _round_trick(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Round trick
        Apply round trick to numerical features
        Args:
            df: polars DataFrame
        Returns:
            polars DataFrame
        Example:
            >>> df = pl.DataFrame({
                "feat_1": [1.23456789, 2.34567890, 3.45678901]
            })
            >>> df = _round_trick(df)
            >>> df
            shape: (3, 1)
        ┌─────────┐
        │ feat_1  │
        │ ---     ┆
        │ f64     ┆
        ╞═════════╡
        │ 1.23    ┆
        │ 2.35    ┆
        │ 3.46    ┆
        └─────────┘
        """
        num_features = [
            col for col in self.cfg.store.num_features if col not in ["seq"]
        ]
        df = df.with_columns(
            [pl.col(col).cast(pl.Float64).round(2).alias(col) for col in num_features]
        )
        return df

    def _get_min_max_id_in_seq_feature(self, df: pl.DataFrame) -> None:
        print("Processing sequence data with Polars operations...")

        # Use Polars operations to process all sequences at once
        result = (
            df.select(self.cfg.data.seq)
            .with_columns(
                [
                    # Split sequences and convert to integers
                    pl.col(self.cfg.data.seq)
                    .str.split(",")
                    .list.eval(
                        pl.element().str.strip_chars().cast(pl.Int64, strict=False)
                    )
                    .alias("seq_list")
                ]
            )
            .select(
                [
                    # Get min and max from all sequences
                    pl.col("seq_list").list.min().min().alias("global_min"),
                    pl.col("seq_list").list.max().max().alias("global_max"),
                ]
            )
        )

        # Extract results
        self.cfg.data.min_id_in_seq = result.select("global_min").item()
        self.cfg.data.max_id_in_seq = result.select("global_max").item()

        print(
            f"Sequence ID range: {self.cfg.data.min_id_in_seq} to {self.cfg.data.max_id_in_seq}"
        )
