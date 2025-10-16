"""
transformer trainer that applies sampling within CV folds to prevent data leakage
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import yaml

from models.transformer import TransformerCTR
from data.dataset import (
    ClickDataset,
    set_random_seeds,
    seq_collate_fn_train,
    seq_collate_fn_infer,
    StratifiedBatchSampler,
)
from data.transformer import preprocess_transformer_data
from data.loader import load_data, apply_negative_sampling
from evaluate.metric import calculate_competition_score
from utils.loss import get_loss_function
from utils.scheduler import get_scheduler
from utils.device import get_device
from utils.logger import setup_logger


class TransformerTrainer:
    """Transformer CTR model trainer that prevents data leakage"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration and setup environment"""
        self.config = config
        self.device = get_device()
        self.logger = setup_logger(
            os.path.join(self.config["train"]["output_dir"], "train_log.log")
        )

        # Setup environment
        self._setup_environment()

        self.logger.info(f"Device: {self.device}")
        self.logger.info("🔧 Using  trainer that prevents data leakage")

    def _setup_environment(self):
        """Setup random seeds and GPU memory management"""
        # Set random seed
        set_random_seeds(self.config["train"]["seed"])

        # GPU memory management
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_limit = total_memory * 0.8
            self.logger.info(f"Total GPU Memory: {total_memory:.2f}GB")
            self.logger.info(f"Memory Limit Set: {memory_limit:.2f}GB (80%)")
            torch.cuda.set_per_process_memory_fraction(0.8, device=0)
            torch.cuda.empty_cache()

    def _load_and_preprocess_data(self) -> Dict[str, Any]:
        """Load and preprocess data WITHOUT applying sampling ( approach)"""
        self.logger.info("데이터 로드 및 전처리 시작 (샘플링 없음 - 올바른 방식)")

        # Load data WITHOUT sampling to prevent data leakage
        train, test = load_data(
            self.config["data"]["path"],
            use_sampling=False,  # CRITICAL: No sampling here!
            sampling_ratio=1.0,
            seed=self.config["data"].get("seed", 42),
        )

        # Preprocess data (also does not apply sampling)
        data_info = preprocess_transformer_data(train, test, self.config, self.logger)

        self.logger.info("데이터 로드 및 전처리 완료 (원본 데이터 유지)")
        self.logger.info(f"원본 학습 데이터 크기: {len(data_info['train']):,}")
        self.logger.info(
            f"원본 양성 비율: {data_info['train'][data_info['target_col']].mean():.4f}"
        )

        return data_info

    def _apply_fold_sampling(
        self, train_fold: pd.DataFrame, data_info: Dict[str, Any], fold_idx: int
    ) -> pd.DataFrame:
        """Apply sampling to a single fold's training data"""
        sampling_config = data_info["sampling_config"]

        if not sampling_config["use_sampling"]:
            return train_fold

        self.logger.info(
            f"Fold {fold_idx}: 샘플링 적용 (비율: {sampling_config['sampling_ratio']:.2f})"
        )

        # Apply sampling with fold-specific seed for reproducibility
        fold_seed = sampling_config["seed"] + fold_idx
        sampled_fold = apply_negative_sampling(
            train_fold.copy(), sampling_config["sampling_ratio"], fold_seed
        )

        self.logger.info(
            f"Fold {fold_idx}: {len(train_fold):,} -> {len(sampled_fold):,} samples"
        )
        self.logger.info(
            f"Fold {fold_idx}: 양성 비율 {train_fold[data_info['target_col']].mean():.4f} -> {sampled_fold[data_info['target_col']].mean():.4f}"
        )

        return sampled_fold

    def _create_data_loaders(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, data_info: Dict[str, Any]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders"""
        # Training dataset and loader
        train_dataset = ClickDataset(
            train_df,
            num_cols=data_info["num_cols"],
            cat_cols=data_info["cat_cols"],
            seq_col=data_info["seq_col"],
            target_col=data_info["target_col"],
            has_target=True,
            max_seq_length=data_info["max_seq_length"],
            seq_split_strategy=self.config["train"].get(
                "seq_split_strategy", "front_random"
            ),
            seq_split_ratio=self.config["train"].get("seq_split_ratio", 0.7),
        )

        # Check if stratified sampling is enabled
        use_stratified_sampling = self.config["train"].get(
            "use_stratified_sampling", False
        )
        positive_ratio = self.config["train"].get("positive_ratio", 0.1)
        batch_size = int(self.config["train"]["batch_size"])

        if use_stratified_sampling:
            self.logger.info(
                f"Stratified sampling 활성화 - 양성 샘플 비율: {positive_ratio:.2f}"
            )
            # Create stratified batch sampler
            train_sampler = StratifiedBatchSampler(
                labels=train_df[data_info["target_col"]].values,
                batch_size=batch_size,
                positive_ratio=positive_ratio,
                shuffle=True,
                seed=self.config["train"].get("seed", 42),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=seq_collate_fn_train,
                pin_memory=True,
                num_workers=2,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=seq_collate_fn_train,
                pin_memory=True,
                num_workers=2,
            )

        # Validation dataset and loader (ALWAYS uses original distribution)
        val_dataset = ClickDataset(
            val_df,
            num_cols=data_info["num_cols"],
            cat_cols=data_info["cat_cols"],
            seq_col=data_info["seq_col"],
            target_col=data_info["target_col"],
            has_target=True,
            max_seq_length=data_info["max_seq_length"],
            seq_split_strategy=self.config["train"].get(
                "seq_split_strategy", "front_random"
            ),
            seq_split_ratio=self.config["train"].get("seq_split_ratio", 0.7),
        )

        # Validation loader - NO stratified sampling to maintain original distribution
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=seq_collate_fn_train,
            pin_memory=True,
            num_workers=2,
        )

        return train_loader, val_loader

    def _create_model_and_optimizer(
        self, train_df: pd.DataFrame, data_info: Dict[str, Any]
    ) -> Tuple[
        TransformerCTR,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        torch.nn.Module,
    ]:
        """Create model, optimizer, scheduler, and loss function"""
        # Create model
        cat_cardinalities = [
            len(data_info["cat_encoders"][c].classes_) for c in data_info["cat_cols"]
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
        ).to(self.device)

        # Loss function - calculate pos_weight from training fold
        pos_weight_value = (
            len(train_df) - train_df[data_info["target_col"]].sum()
        ) / train_df[data_info["target_col"]].sum()
        self.logger.info(f"Pos weight value: {pos_weight_value:.4f}")

        # Get loss function parameters from config
        loss_params = self.config["train"].get("loss_params", {})
        loss_params["pos_weight"] = pos_weight_value

        criterion = get_loss_function(
            self.config["train"]["loss_function"], **loss_params
        )

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config["train"]["learning_rate"]),
            weight_decay=float(self.config["train"]["weight_decay"]),
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Learning rate scheduler
        scheduler_params = self.config["train"].get("scheduler_params", {})
        scheduler = get_scheduler(
            self.config["train"].get("scheduler", "reduce_on_plateau"),
            optimizer,
            int(self.config["train"]["epochs"]),
            **scheduler_params,
        )

        return model, optimizer, scheduler, criterion

    def _train_epoch(
        self,
        model: TransformerCTR,
        train_loader: DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        gradient_accumulation_steps = int(
            self.config["train"]["gradient_accumulation_steps"]
        )

        for batch_idx, (num_x, cat_x, seqs, lens, ys) in enumerate(
            tqdm(train_loader, desc=f"[Train Epoch {epoch}]")
        ):
            num_x, cat_x, seqs, lens, ys = (
                num_x.to(self.device),
                cat_x.to(self.device),
                seqs.to(self.device),
                lens.to(self.device),
                ys.to(self.device),
            )

            # Input validation
            if (
                torch.any(torch.isnan(num_x))
                or torch.any(torch.isnan(cat_x))
                or torch.any(torch.isnan(seqs))
            ):
                self.logger.error(
                    f"Epoch {epoch}, Batch {batch_idx}: 입력 데이터에 NaN 발견!"
                )
                continue

            # Forward pass
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys) / gradient_accumulation_steps

            # Loss validation
            if torch.isnan(loss) or torch.isinf(loss):
                self.logger.error(f"Epoch {epoch}, Batch {batch_idx}: Loss가 NaN/Inf!")
                continue

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.error(
                        f"Epoch {epoch}, Batch {batch_idx}: Gradient가 NaN/Inf!"
                    )
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ys.size(0) * gradient_accumulation_steps

            # Memory cleanup
            if (
                batch_idx % int(self.config["train"]["memory_cleanup_frequency"]) == 0
                and torch.cuda.is_available()
            ):
                torch.cuda.empty_cache()

        return total_loss / len(train_loader.dataset)

    def _validate_epoch(
        self,
        model: TransformerCTR,
        val_loader: DataLoader,
        criterion: torch.nn.Module,
        data_info: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Validate model for one epoch and return loss and competition score"""
        model.eval()
        val_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for num_x, cat_x, seqs, lens, ys in val_loader:
                num_x, cat_x, seqs, lens, ys = (
                    num_x.to(self.device),
                    cat_x.to(self.device),
                    seqs.to(self.device),
                    lens.to(self.device),
                    ys.to(self.device),
                )

                logits = model(num_x, cat_x, seqs, lens)
                loss = criterion(logits, ys)
                val_loss += loss.item() * ys.size(0)

                # Store predictions and targets for competition score
                preds = torch.sigmoid(logits).cpu().numpy()
                targets = ys.cpu().numpy()
                all_predictions.extend(preds)
                all_targets.extend(targets)

        # Calculate average loss
        avg_loss = val_loss / len(val_loader.dataset)

        # Calculate competition score
        try:
            competition_score, ap_score, wll_score = calculate_competition_score(
                np.array(all_targets), np.array(all_predictions)
            )
        except Exception as e:
            self.logger.warning(f"Failed to calculate competition score: {e}")
            competition_score = 0.0

        return avg_loss, competition_score

    def train_model(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, data_info: Dict[str, Any]
    ) -> TransformerCTR:
        """Train transformer model"""
        self.logger.info("모델 학습 시작")

        # Step 1: Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            train_df, val_df, data_info
        )

        # Step 2: Create model and optimizer
        model, optimizer, scheduler, criterion = self._create_model_and_optimizer(
            train_df, data_info
        )

        # Step 3: Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, int(self.config["train"]["epochs"]) + 1):
            # Set epoch for stratified sampler (important for proper shuffling)
            if hasattr(train_loader, "batch_sampler") and hasattr(
                train_loader.batch_sampler, "set_epoch"
            ):
                train_loader.batch_sampler.set_epoch(epoch)

            # Train one epoch
            train_loss = self._train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )

            # Validate one epoch
            val_loss, competition_score = self._validate_epoch(
                model, val_loader, criterion, data_info
            )

            # Get current learning rate
            current_lr = optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Comp Score: {competition_score:.4f}, LR: {current_lr:.6f}"
            )

            # Learning rate scheduler step
            if (
                hasattr(scheduler, "step")
                and scheduler.__class__.__name__ == "ReduceLROnPlateau"
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step(epoch)

            # Early stopping check - use competition score if available
            metric_to_track = (
                competition_score if competition_score > 0 else -val_loss
            )  # Higher is better for competition score
            if metric_to_track > best_val_loss - float(
                self.config["train"]["min_delta"]
            ):
                best_val_loss = metric_to_track
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= int(
                    self.config["train"]["early_stopping_patience"]
                ):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info("모델 학습 완료")
        return model

    def _create_kfold_splitter(self) -> StratifiedKFold:
        """Create K-Fold splitter"""
        return StratifiedKFold(
            n_splits=int(self.config["train"]["n_folds"]),
            shuffle=True,
            random_state=int(self.config["train"]["seed"]),
        )

    def _train_fold(
        self,
        fold: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        train: pd.DataFrame,
        data_info: Dict[str, Any],
    ) -> Tuple[TransformerCTR, np.ndarray]:
        """Train one fold with  sampling approach"""
        self.logger.info(f"=== Fold {fold}/{int(self.config['train']['n_folds'])} ===")

        # Data split - use original data for validation
        train_fold_orig = train.iloc[train_idx].reset_index(drop=True)
        val_fold = train.iloc[val_idx].reset_index(
            drop=True
        )  # Keep original distribution

        # Apply sampling ONLY to training fold
        train_fold = self._apply_fold_sampling(train_fold_orig, data_info, fold)

        self.logger.info(
            f"Train size: {len(train_fold):,}, Val size: {len(val_fold):,}"
        )
        self.logger.info(
            f"Train pos rate: {train_fold[data_info['target_col']].mean():.4f}"
        )
        self.logger.info(
            f"Val pos rate: {val_fold[data_info['target_col']].mean():.4f} (original distribution)"
        )

        # Train model
        model = self.train_model(train_fold, val_fold, data_info)

        # OOF prediction on original validation distribution
        val_predictions = self._get_oof_predictions(model, val_fold, data_info)

        # Fold performance
        fold_score, fold_ap, fold_wll = calculate_competition_score(
            val_fold[data_info["target_col"]].values, val_predictions
        )
        self.logger.info(
            f"Fold {fold} - Competition Score: {fold_score:.4f}, AP: {fold_ap:.4f}, WLL: {fold_wll:.4f}"
        )

        # Memory cleanup
        del train_fold_orig, train_fold, val_fold
        torch.cuda.empty_cache()

        return model, val_predictions

    def _get_oof_predictions(
        self, model: TransformerCTR, val_fold: pd.DataFrame, data_info: Dict[str, Any]
    ) -> np.ndarray:
        """Get out-of-fold predictions for validation set"""
        val_dataset = ClickDataset(
            val_fold,
            num_cols=data_info["num_cols"],
            cat_cols=data_info["cat_cols"],
            seq_col=data_info["seq_col"],
            target_col=data_info["target_col"],
            has_target=True,
            max_seq_length=data_info["max_seq_length"],
            seq_split_strategy=self.config["train"].get(
                "seq_split_strategy", "front_random"
            ),
            seq_split_ratio=self.config["train"].get("seq_split_ratio", 0.7),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(self.config["train"]["batch_size"]),
            shuffle=False,
            collate_fn=seq_collate_fn_train,
            pin_memory=True,
        )

        model.eval()
        val_predictions = []
        with torch.no_grad():
            for num_x, cat_x, seqs, lens, ys in val_loader:
                num_x, cat_x, seqs, lens = (
                    num_x.to(self.device),
                    cat_x.to(self.device),
                    seqs.to(self.device),
                    lens.to(self.device),
                )

                logits = model(num_x, cat_x, seqs, lens)
                preds = torch.sigmoid(logits).cpu()
                val_predictions.append(preds)

        return torch.cat(val_predictions).numpy()

    def run_cv_training(
        self, data_info: Dict[str, Any]
    ) -> tuple[List[TransformerCTR], np.ndarray]:
        """Run cross-validation training with  approach"""
        self.logger.info("K-Fold 교차 검증 학습 시작 (올바른 방식)")

        train = data_info["train"]  # Original train data
        target_col = data_info["target_col"]

        # Step 1: Create K-Fold splitter
        kfold = self._create_kfold_splitter()

        # Step 2: Initialize results
        models = []
        oof_predictions = np.zeros(len(train))

        # Step 3: Train each fold with  sampling
        for fold, (train_idx, val_idx) in enumerate(
            kfold.split(train, train[target_col]), 1
        ):
            model, val_predictions = self._train_fold(
                fold, train_idx, val_idx, train, data_info
            )

            # Store results
            oof_predictions[val_idx] = val_predictions
            models.append(model)

        # Step 4: Calculate overall OOF performance on original distribution
        oof_score, oof_ap, oof_wll = calculate_competition_score(
            train[target_col].values, oof_predictions
        )
        self.logger.info(
            f"전체 OOF 성능 (원본 분포) - Competition Score: {oof_score:.4f}, AP: {oof_ap:.4f}, WLL: {oof_wll:.4f}"
        )

        self.logger.info("K-Fold 교차 검증 완료!")
        return models, oof_predictions

    def _create_test_dataset(
        self, test: pd.DataFrame, data_info: Dict[str, Any]
    ) -> DataLoader:
        """Create test dataset and loader"""
        test_dataset = ClickDataset(
            test,
            num_cols=data_info["num_cols"],
            cat_cols=data_info["cat_cols"],
            seq_col=data_info["seq_col"],
            has_target=False,
            max_seq_length=data_info["max_seq_length"],
            seq_split_strategy=self.config["train"].get(
                "seq_split_strategy", "front_random"
            ),
            seq_split_ratio=self.config["train"].get("seq_split_ratio", 0.7),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(self.config["train"]["batch_size"]),
            shuffle=False,
            collate_fn=seq_collate_fn_infer,
            pin_memory=True,
        )
        return test_loader

    def _predict_with_model(
        self, model: TransformerCTR, test_loader: DataLoader, fold_idx: int
    ) -> np.ndarray:
        """Predict with single model"""
        self.logger.info(f"Fold {fold_idx} 모델로 추론 중...")
        model.eval()
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

        return torch.cat(fold_predictions).numpy()

    def predict_test(
        self, models: List[TransformerCTR], data_info: Dict[str, Any]
    ) -> np.ndarray:
        """Predict on test data with ensemble"""
        self.logger.info("앙상블 추론 시작")

        test = data_info["test"]

        # Step 1: Create test dataset
        test_loader = self._create_test_dataset(test, data_info)

        # Step 2: Get predictions from all models
        all_predictions = []
        for fold_idx, model in enumerate(models, 1):
            fold_predictions = self._predict_with_model(model, test_loader, fold_idx)
            all_predictions.append(fold_predictions)

        # Step 3: Ensemble prediction (average)
        test_preds = np.mean(all_predictions, axis=0)
        self.logger.info("앙상블 추론 완료")

        return test_preds

    def run(self):
        """Main training pipeline with  approach"""
        start_time = time.time()

        # Step 1: Load and preprocess data (without sampling)
        data_info = self._load_and_preprocess_data()

        # Step 2: Run cross-validation training (with proper sampling)
        models, oof_predictions = self.run_cv_training(data_info)

        # Step 3: Predict on test data
        test_preds = self.predict_test(models, data_info)

        # Step 4: Save results
        self.save_results(test_preds, data_info)

        # Step 5: Log execution time
        total_time = time.time() - start_time
        self.logger.info(
            f"전체 실행 시간: {total_time / 3600:.2f}시간 ({total_time:.2f}초)"
        )

    def save_results(self, test_preds: np.ndarray, data_info: Dict[str, Any]):
        """Save prediction results"""
        test = data_info["test"]
        id_col = data_info["id_col"]

        # Create submission file
        submit = pd.DataFrame({"ID": test[id_col].values, "clicked": test_preds})

        # Save submission
        output_dir = Path(self.config["train"]["output_dir"])
        output_dir.mkdir(exist_ok=True)

        timestamp = int(time.time())
        submit_filename = output_dir / f"transformer_submission_{timestamp}.csv"
        submit.to_csv(submit_filename, index=False)

        self.logger.info(f"제출 파일 저장 완료: {submit_filename}")


def main():
    """Main function"""
    # Load configuration from unified YAML file
    config_path = Path(__file__).parent.parent / "config" / "transformer.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize  trainer
    trainer = TransformerTrainer(config)

    # Run training
    trainer.run()


if __name__ == "__main__":
    main()
