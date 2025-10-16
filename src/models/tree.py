from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from omegaconf import OmegaConf
from typing_extensions import Self

from evaluate.metric import (
    CompetitionScoreMetric,
    calculate_competition_score,
    calculate_weighted_logloss,
)
from models.base import BaseModel


class LightGBMTrainer(BaseModel):
    def __init__(
        self,
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
        super().__init__(
            model_path,
            results,
            params,
            early_stopping_rounds,
            num_boost_round,
            verbose_eval,
            seed,
            features,
            cat_features,
            n_splits,
            split_type,
            logger,
        )

    def _weighted_logloss_lgb(
        self, y_pred: np.ndarray, y_true: lgb.Dataset
    ) -> tuple[str, float, bool]:
        """LightGBM custom metric for Weighted LogLoss with 50:50 class weights"""
        y_true = y_true.get_label()
        wll = calculate_weighted_logloss(y_true, y_pred)
        return "weighted_logloss", wll, False

    def _competition_score_lgb(
        self, y_pred: np.ndarray, y_true: lgb.Dataset
    ) -> tuple[str, float, bool]:
        """LightGBM custom metric for competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
        y_true = y_true.get_label()
        score, _, _ = calculate_competition_score(y_true, y_pred)
        return "competition_score", score, True  # True means higher is better

    def _fit(
        self: Self,
        X_train: pl.DataFrame | np.ndarray,
        y_train: pl.Series | np.ndarray,
        X_valid: pl.DataFrame | np.ndarray | None = None,
        y_valid: pl.Series | np.ndarray | None = None,
    ) -> lgb.Booster:
        X_train, y_train = X_train[self.features].to_pandas(), y_train.to_pandas()
        X_valid, y_valid = X_valid[self.features].to_pandas(), y_valid.to_pandas()

        # set params
        params = OmegaConf.to_container(self.params)
        params["seed"] = self.seed

        train_set = lgb.Dataset(
            X_train,
            y_train,
            params=params,
            categorical_feature=self.cat_features,
            feature_name=self.features,
        )
        valid_set = lgb.Dataset(
            X_valid,
            y_valid,
            params=params,
            categorical_feature=self.cat_features,
            feature_name=self.features,
        )

        # dart boosting의 경우 early_stopping 사용하지 않음 (내부적으로 처리불가하여 콜백으로 처리)
        callbacks = (
            [
                lgb.log_evaluation(self.verbose_eval),
                lgb.early_stopping(self.early_stopping_rounds),
            ]
            if params.get("boosting_type") != "dart"
            else [lgb.log_evaluation(self.verbose_eval)]
        )

        model = lgb.train(
            params=params,
            train_set=train_set,
            valid_sets=[valid_set],
            num_boost_round=self.num_boost_round,
            feval=self._competition_score_lgb,
            callbacks=callbacks,
        )

        return model

    def _predict(
        self: Self, model: lgb.Booster, X: pl.DataFrame | np.ndarray
    ) -> np.ndarray:
        return model.predict(X[self.features].to_pandas())

    def load_model(self: Self) -> lgb.Booster:
        return lgb.Booster(model_file=Path(self.model_path) / f"{self.results}.model")


class XGBoostTrainer(BaseModel):
    def __init__(
        self,
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
        super().__init__(
            model_path,
            results,
            params,
            early_stopping_rounds,
            num_boost_round,
            verbose_eval,
            seed,
            features,
            cat_features,
            n_splits,
            split_type,
            logger,
        )

    def _weighted_logloss_xgb(
        self, y_pred: np.ndarray, y_true: xgb.DMatrix
    ) -> tuple[str, float]:
        """XGBoost custom metric for Weighted LogLoss with 50:50 class weights"""
        y_true = y_true.get_label()
        wll = calculate_weighted_logloss(y_true, y_pred)
        return "weighted_logloss", wll

    def _competition_score_xgb(
        self, y_pred: np.ndarray, y_true: xgb.DMatrix
    ) -> tuple[str, float]:
        """XGBoost custom metric for competition score: 0.5*AP + 0.5*(1/(1+WLL))"""
        y_true = y_true.get_label()
        score, _, _ = calculate_competition_score(y_true, y_pred)
        return "competition_score", score

    def _fit(
        self: Self,
        X_train: pl.DataFrame | np.ndarray,
        y_train: pl.Series | np.ndarray,
        X_valid: pl.DataFrame | np.ndarray | None = None,
        y_valid: pl.Series | np.ndarray | None = None,
    ) -> xgb.Booster:
        params = OmegaConf.to_container(self.params)
        params["seed"] = self.seed

        X_train, y_train = X_train[self.features].to_pandas(), y_train.to_pandas()
        X_valid, y_valid = X_valid[self.features].to_pandas(), y_valid.to_pandas()

        train_set = xgb.DMatrix(X_train, y_train)
        valid_set = xgb.DMatrix(X_valid, y_valid)

        model = xgb.train(
            params=params,
            dtrain=train_set,
            num_boost_round=self.num_boost_round,
            evals=[(valid_set, "valid")],
            custom_metric=self._competition_score_xgb,
            maximize=True,
            early_stopping_rounds=self.early_stopping_rounds
            if params["booster"] != "dart"
            else None,
            verbose_eval=self.verbose_eval,
        )

        return model

    def _predict(
        self: Self, model: xgb.Booster, X_test: pl.DataFrame | np.ndarray
    ) -> np.ndarray:
        X_test = X_test[self.features].to_pandas()
        return model.predict(xgb.DMatrix(X_test))

    def load_model(self: Self) -> xgb.Booster:
        return xgb.Booster(model_file=Path(self.model_path) / f"{self.results}.json")


class CatBoostTrainer(BaseModel):
    def __init__(
        self,
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
        use_encoder: bool = False,
        logger: logging.Logger = None,
    ) -> None:
        super().__init__(
            model_path,
            results,
            params,
            early_stopping_rounds,
            num_boost_round,
            verbose_eval,
            seed,
            features,
            cat_features,
            n_splits,
            split_type,
            logger,
        )
        self.use_encoder = use_encoder
        self.encoder = None

    def _fit(
        self: Self,
        X_train: pl.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pl.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> CatBoostClassifier:
        X_train, y_train = X_train[self.features].to_pandas(), y_train.to_pandas()
        X_valid, y_valid = X_valid[self.features].to_pandas(), y_valid.to_pandas()

        train_set = Pool(
            X_train,
            y_train,
            cat_features=self.cat_features,
        )
        valid_set = Pool(
            X_valid,
            y_valid,
            cat_features=self.cat_features,
        )

        params = OmegaConf.to_container(self.params)
        params["random_seed"] = self.seed

        model = CatBoostClassifier(
            **params,
            iterations=self.num_boost_round,
            verbose=self.verbose_eval,
            eval_metric=CompetitionScoreMetric(),
        )

        model.fit(
            train_set,
            eval_set=valid_set,
            verbose_eval=self.verbose_eval,
            early_stopping_rounds=self.early_stopping_rounds,
        )

        return model

    def _predict(
        self: Self, model: CatBoostClassifier, X_test: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        X_test = X_test[self.features].to_pandas()
        return model.predict_proba(Pool(X_test, cat_features=self.cat_features))[:, 1]

    def load_model(self: Self) -> CatBoostClassifier:
        return CatBoostClassifier().load_model(
            Path(self.model_path) / f"{self.results}.cbm"
        )
