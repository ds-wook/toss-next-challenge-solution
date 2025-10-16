from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from omegaconf import DictConfig
from tqdm import tqdm

from data.tree import TreeDataLoader


def inference_models(
    cfg: DictConfig, test_x: pd.DataFrame | dict[str, pd.Series]
) -> np.ndarray:
    """Given a model, predict probabilities for each class.
    Args:
        results: ModelResult object
        test_x: test dataframe
    Returns:
        predict probabilities for each class
    """
    # load model
    results = joblib.load(Path(cfg.models.model_path) / f"{cfg.models.results}.pkl")
    folds = len(results.models)
    preds = np.zeros((test_x.shape[0],))

    for model in tqdm(
        results.models.values(), total=folds, desc="Predicting models", colour="blue"
    ):
        if isinstance(model, xgb.Booster):
            features = [
                *cfg.store.num_features,
                *cfg.store.cat_features,
                *cfg.store.count_features,
            ]
            preds += model.predict(xgb.DMatrix(test_x[features])) / folds

        elif isinstance(model, CatBoostClassifier):
            features = [*cfg.store.num_features, *cfg.store.cat_features]
            preds += (
                model.predict_proba(
                    Pool(
                        test_x[features],
                        cat_features=cfg.store.cat_features,
                    )
                )[:, 1]
                / folds
            )

        else:
            features = [*cfg.store.num_features, *cfg.store.cat_features]
            preds += model.predict(test_x[features]) / folds

    return preds


@hydra.main(config_path="../config/", config_name="predict", version_base="1.3.1")
def _main(cfg: DictConfig):
    # load test dataset
    data_loader = TreeDataLoader(cfg)

    # load count mapping before loading test data
    count_mapping_path = Path(cfg.data.encoder_path) / "count_mapping.pkl"
    data_loader.load_count_mapping(count_mapping_path)

    test_x = data_loader.load_test_data(is_boosting=True)
    test_x = test_x.to_pandas()
    # load submit dataset
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

    # predict
    preds = inference_models(cfg, test_x)
    submit[cfg.data.target] = preds
    submit.to_csv(Path(cfg.output.path) / f"{cfg.models.results}.csv", index=False)


if __name__ == "__main__":
    _main()
