import logging
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.transformer import TransformerCTR
from data.dataset import ClickDataset, seq_collate_fn_infer
from data.transformer import preprocess_transformer_data
from data.loader import load_data

# Direct imports to avoid dependency issues
try:
    from utils.device import get_device
except ImportError:

    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


try:
    from utils.logger import setup_logger
except ImportError:

    def setup_logger(name: str):
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


class TransformerPredictor:
    """Transformer CTR model predictor"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = get_device()
        self.logger = setup_logger("transformer_predictor")

        self.logger.info(f"Device: {self.device}")

    def load_data(self) -> pd.DataFrame:
        """Load test data using existing loader utility"""
        data_path = str(self.config["data"]["path"])

        # Use existing load_data function and extract test data
        _, test = load_data(data_path)

        self.logger.info(f"Test shape: {test.shape}")
        return test

    def preprocess_data(self, test: pd.DataFrame) -> tuple:
        """Preprocess test data using existing utilities"""
        # Load training data for preprocessing (to get encoders and scalers)
        data_path = str(self.config["data"]["path"])
        train, _ = load_data(data_path)

        # Use the complete preprocessing pipeline
        return preprocess_transformer_data(train, test, self.config, self.logger)

    def load_models(self, data_info: Dict[str, Any]) -> List[TransformerCTR]:
        """Load trained models"""
        model_path = Path(self.config["predict"]["model_path"])

        if not model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")

        models = []

        # Load all fold models
        for fold_idx in range(1, int(self.config["predict"]["n_folds"]) + 1):
            model_file = model_path / f"transformer_fold_{fold_idx}.pth"

            if not model_file.exists():
                self.logger.warning(
                    f"Model file {model_file} not found, skipping fold {fold_idx}"
                )
                continue

            # Create model instance
            cat_cardinalities = [
                len(data_info["cat_encoders"][c].classes_)
                for c in data_info["cat_cols"]
            ]
            model = TransformerCTR(
                num_features=len(data_info["num_cols"]),
                cat_cardinalities=cat_cardinalities,
                emb_dim=self.config["model"]["emb_dim"],
                transformer_dim=self.config["model"]["transformer_dim"],
                transformer_heads=self.config["model"]["transformer_heads"],
                transformer_layers=self.config["model"]["transformer_layers"],
                hidden_units=self.config["model"]["hidden_units"],
                dropout=self.config["model"]["dropout"],
            )

            # Load model weights
            checkpoint = torch.load(model_file, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            models.append(model)
            self.logger.info(f"Loaded model for fold {fold_idx}")

        if not models:
            raise ValueError("No models loaded successfully")

        self.logger.info(f"Loaded {len(models)} models")
        return models

    def predict(
        self, models: List[TransformerCTR], data_info: Dict[str, Any]
    ) -> np.ndarray:
        """Predict on test data"""
        test = data_info["test"]

        test_dataset = ClickDataset(
            test,
            num_cols=data_info["num_cols"],
            cat_cols=data_info["cat_cols"],
            seq_col=data_info["seq_col"],
            has_target=False,
            max_seq_length=data_info["max_seq_length"],
            seq_split_strategy=self.config["predict"].get(
                "seq_split_strategy", "front_random"
            ),
            seq_split_ratio=self.config["predict"].get("seq_split_ratio", 0.7),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(self.config["predict"]["batch_size"]),
            shuffle=False,
            collate_fn=seq_collate_fn_infer,
            pin_memory=True,
        )

        self.logger.info("앙상블 추론 시작")
        all_predictions = []

        for fold_idx, model in enumerate(models, 1):
            self.logger.info(f"Fold {fold_idx} 모델로 추론 중...")
            fold_predictions = []

            with torch.no_grad():
                for num_x, cat_x, seqs, lens in tqdm(
                    test_loader, desc=f"[Fold {fold_idx} Inference]"
                ):
                    num_x, cat_x, seqs, lens = (
                        num_x.to(self.device),
                        cat_x.to(self.device),
                        seqs.to(self.device),
                        lens.to(self.device),
                    )

                    logits = model(num_x, cat_x, seqs, lens)
                    preds = torch.sigmoid(logits).cpu()
                    fold_predictions.append(preds)

            fold_predictions = torch.cat(fold_predictions).numpy()
            all_predictions.append(fold_predictions)

        # Ensemble prediction (average)
        test_preds = np.mean(all_predictions, axis=0)
        self.logger.info("앙상블 추론 완료")

        return test_preds

    def save_predictions(self, test_preds: np.ndarray, data_info: Dict[str, Any]):
        """Save prediction results"""
        test = data_info["test"]
        id_col = data_info["id_col"]

        # Create submission file
        submit = pd.DataFrame({"ID": test[id_col].values, "clicked": test_preds})

        # Save submission
        output_dir = Path(self.config["predict"]["output_dir"])
        output_dir.mkdir(exist_ok=True)

        import time

        timestamp = int(time.time())
        submit_filename = output_dir / f"transformer_prediction_{timestamp}.csv"
        submit.to_csv(submit_filename, index=False)

        self.logger.info(f"예측 결과 저장 완료: {submit_filename}")

        # Log prediction statistics
        self.logger.info("예측값 통계:")
        self.logger.info(f"  - 평균: {test_preds.mean():.6f}")
        self.logger.info(f"  - 표준편차: {test_preds.std():.6f}")
        self.logger.info(f"  - 최솟값: {test_preds.min():.6f}")
        self.logger.info(f"  - 최댓값: {test_preds.max():.6f}")

    def run(self):
        """Main prediction pipeline"""
        # Load test data
        test = self.load_data()

        # Preprocess data
        data_info = self.preprocess_data(test)

        # Load models
        models = self.load_models(data_info)

        # Predict
        test_preds = self.predict(models, data_info)

        # Save results
        self.save_predictions(test_preds, data_info)


def main():
    """Main function"""
    # Load configuration from unified YAML file
    config_path = Path(__file__).parent.parent / "config" / "transformer.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize predictor with loaded config
    predictor = TransformerPredictor(config)

    # Run prediction
    predictor.run()


if __name__ == "__main__":
    main()
