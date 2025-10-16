from __future__ import annotations

import gc
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from tqdm import tqdm
from typing_extensions import Self

from evaluate.metric import calculate_competition_score
from models.cv import LeaveOneDayOutCV


@dataclass
class ModelResult:
    oof_preds: np.ndarray
    models: dict[str, Any]


class BaseModel(ABC):
    def __init__(
        self: Self,
        model_path: str,
        results: str,
        params: dict[str, Any],
        early_stopping_rounds: int,
        num_boost_round: int,
        verbose_eval: int,
        seed: int,
        features: list[str],
        cat_features: list[str],
        n_splits: int = 5,
        split_type: str = "day_of_week",
        logger: logging.Logger = None,
    ) -> None:
        self.model_path = model_path
        self.results = results
        self.params = params
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.verbose_eval = verbose_eval
        self.seed = seed
        self.model = None
        self.features = features
        self.cat_features = cat_features
        self.n_splits = n_splits
        self.split_type = split_type
        self.logger = logger

    def save_model(self, save_dir: Path) -> None:
        joblib.dump(self.result, save_dir)

    @abstractmethod
    def load_model(self: Self):
        # return model
        raise NotImplementedError

    @abstractmethod
    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        raise NotImplementedError

    def fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ):
        model = self._fit(X_train, y_train, X_valid, y_valid)

        return model

    @abstractmethod
    def _predict(self: Self, model: Any, X: pd.DataFrame | np.ndarray):
        raise NotImplementedError

    def run_cv_training(
        self: Self, X: pl.DataFrame, y: pl.Series, groups: pl.Series | None = None
    ) -> Self:
        oof_preds = np.zeros(X.shape[0])
        models = {}

        match self.split_type:
            case "day_of_week":
                kfold = LeaveOneDayOutCV(day_col="day_of_week")
                k_splits = kfold.split(X)
            case "group":
                kfold = StratifiedGroupKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.seed
                )
                k_splits = kfold.split(X, y, groups=groups)
            case _:
                kfold = StratifiedKFold(
                    n_splits=self.n_splits, shuffle=True, random_state=self.seed
                )
                k_splits = kfold.split(X, y)

        with tqdm(k_splits, total=kfold.get_n_splits(X, y)) as pbar:
            for fold, (train_idx, valid_idx) in enumerate(pbar, 1):
                X_train, X_valid = X[train_idx], X[valid_idx]
                y_train, y_valid = y[train_idx], y[valid_idx]

                if "xgboost" in self.results:
                    X_train, X_valid = self._encode_categorical_count(X_train, X_valid)

                model = self.fit(X_train, y_train, X_valid, y_valid)
                oof_preds[valid_idx] = self._predict(model, X_valid)
                models[f"fold_{fold}"] = model
                score, ap, wll = calculate_competition_score(
                    y_valid, oof_preds[valid_idx]
                )
                self.logger.info(
                    f"Fold {fold} - Competition Score: {score:.6f}, AP: {ap:.6f}, WLL: {wll:.6f}"
                )
                del X_train, X_valid, y_train, y_valid, model
                gc.collect()

        score, ap, wll = calculate_competition_score(y, oof_preds)
        self.logger.info(f"Competition score: {score}, AP: {ap}, WLL: {wll}")

        self.result = ModelResult(oof_preds=oof_preds, models=models)

        return self

    def _encode_categorical_count(
        self, X_train: pl.DataFrame, X_valid: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        for col in self.cat_features:
            # Train: count 계산 후 저장
            count_map = (
                X_train.group_by(col)
                .agg(pl.count().alias("count"))
                .select([pl.col(col), pl.col("count")])
            )
            X_train = X_train.join(
                count_map.rename({"count": f"{col}_count"}), on=col, how="left"
            )
            X_valid = X_valid.join(
                count_map.rename({"count": f"{col}_count"}), on=col, how="left"
            )
        return X_train, X_valid
