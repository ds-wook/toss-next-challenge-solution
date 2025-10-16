from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import rankdata
from tqdm import tqdm


def ensemble_predictions(
    predictions: list[np.ndarray], weights: list[float] = None, method: str = "linear"
) -> np.ndarray:
    """
    Ensemble predictions using different methods
    Args:
        predictions: list of predictions
        weights: list of weights
        method: method to ensemble predictions
    Returns:
        ensemble predictions
    """
    if weights is None:
        weights = [1.0 / len(predictions)] * len(predictions)

    assert np.isclose(np.sum(weights), 1.0)

    match method:
        case "linear":
            res = np.average(predictions, weights=weights, axis=0)

        case "harmonic":
            res = np.average([1 / p for p in predictions], weights=weights, axis=0)
            return 1 / res

        case "geometric":
            numerator = np.average(
                [np.log(p) for p in predictions], weights=weights, axis=0
            )
            res = np.exp(numerator)

        case "rank":
            res = np.average(
                [rankdata(p) for p in predictions], weights=weights, axis=0
            )
            return res / (len(res) + 1)

        case "sigmoid":
            # Convert predictions to numpy arrays for element-wise operations
            pred_arrays = [np.asarray(p) for p in predictions]
            eps = 1e-15
            logit_values = [
                np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
                for p in pred_arrays
            ]
            result = np.average(logit_values, weights=weights, axis=0)

            return 1 / (1 + np.exp(-result))

        case _:
            raise ValueError(f"Unknown ensemble method: {method}")

    return res


@hydra.main(config_path="../config/", config_name="ensemble", version_base="1.3.1")
def _main(cfg: DictConfig):
    # Load submission file
    submit = pd.read_csv(Path(cfg.data.path) / f"{cfg.data.submit}.csv")

    # Load predictions and calculate ranks
    preds = [
        pd.read_csv(Path(cfg.output.path) / f"{pred}.csv")[cfg.data.target].to_numpy()
        for pred in tqdm(
            cfg.blends.preds,
            desc="Loading predictions",
            colour="red",
            total=len(cfg.blends.preds),
        )
    ]

    # Calculate average predictions
    submit[cfg.data.target] = ensemble_predictions(
        preds, cfg.blends.weights, cfg.blends.method
    )

    # Save the ensembled submission
    submit.to_csv(Path(cfg.output.path) / f"{cfg.output.name}.csv", index=False)


if __name__ == "__main__":
    _main()
