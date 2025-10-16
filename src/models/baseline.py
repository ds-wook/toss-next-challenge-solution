"""
CTR prediction model using tabular and sequence features
"""

import os
import random
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Self

from data.dataset import ClickDataset, collate_fn_train
from models.base import BaseModel


class TabularSeqModel(nn.Module):
    """
    Tabular + Sequence model for CTR prediction

    This model combines tabular features with sequence data for click-through rate prediction.
    It uses:
    - Batch normalization for tabular features
    - LSTM for processing variable-length sequences
    - MLP for final prediction

    Architecture:
        1. Tabular features -> BatchNorm -> MLP input
        2. Sequence data -> LSTM -> final hidden state
        3. Concatenate tabular + sequence features
        4. MLP layers with dropout for final prediction
    """

    def __init__(
        self,
        d_features: int,
        lstm_hidden: int = 32,
        hidden_units: List[int] = [1024, 512, 256, 128],
        dropout: float = 0.2,
    ):
        """
        Initialize the TabularSeqModel

        Args:
            d_features: Number of tabular features (input dimension)
            lstm_hidden: LSTM hidden state size for sequence processing
            hidden_units: List of hidden layer sizes for the MLP component
            dropout: Dropout rate for regularization (applied to MLP layers)
        """
        super().__init__()

        # Batch normalization for tabular features
        self.bn_x = nn.BatchNorm1d(d_features)

        # LSTM for sequence data
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        # MLP layers
        input_dim = d_features + lstm_hidden
        layers = []

        for h in hidden_units:
            layers.extend([nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            input_dim = h

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, x_feats: torch.Tensor, x_seq: torch.Tensor, seq_lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model

        Args:
            x_feats: Tabular features tensor of shape (B, d_features)
            x_seq: Padded sequence data tensor of shape (B, L) where L is max sequence length
            seq_lengths: Actual sequence lengths tensor of shape (B,) for each sample

        Returns:
            Logits tensor of shape (B,) representing CTR predictions
        """
        # Tabular features
        x = self.bn_x(x_feats)

        # Sequence processing
        x_seq = x_seq.unsqueeze(-1)  # (B, L, 1)

        # Pack sequence for LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]  # (B, lstm_hidden)

        # Combine features
        z = torch.cat([x, h], dim=1)
        return self.mlp(z).squeeze(1)  # logits


def create_model(
    d_features: int,
    lstm_hidden: int = 64,
    hidden_units: List[int] = [256, 128],
    dropout: float = 0.2,
    device: str = "cuda",
    model_type: str = "baseline",
    **kwargs,
) -> nn.Module:
    """
    Create and initialize CTR prediction model

    Args:
        d_features: Number of tabular features
        lstm_hidden: LSTM hidden size for sequence processing
        hidden_units: MLP hidden layer sizes
        dropout: Dropout rate for regularization
        device: Device to place model on ('cuda' or 'cpu')
        model_type: Type of model to create ('baseline', 'deepfm', 'xdeepfm', 'din', 'transformer')
        **kwargs: Additional model parameters

    Returns:
        Initialized PyTorch model ready for training

    Raises:
        ValueError: If unsupported model_type is provided
    """
    if model_type == "baseline":
        model = TabularSeqModel(
            d_features=d_features,
            lstm_hidden=lstm_hidden,
            hidden_units=hidden_units,
            dropout=dropout,
        )
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types: ['baseline']"
        )

    return model.to(device)


def seed_everything(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaselineTrainer(BaseModel):
    """
    Baseline CTR model trainer using TabularSeqModel
    Integrates with the project's BaseModel structure
    """

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
        )
        self.device = params.get("device", "cuda")
        self.model = None

    def _fit(
        self: Self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_valid: pd.DataFrame | np.ndarray | None = None,
        y_valid: pd.Series | np.ndarray | None = None,
    ) -> TabularSeqModel:
        """Train the baseline model"""
        # Set random seed
        seed_everything(self.seed)

        # Convert to tensors and create data loaders
        train_loader = self._create_data_loader(X_train, y_train, is_training=True)
        valid_loader = (
            self._create_data_loader(X_valid, y_valid, is_training=False)
            if X_valid is not None
            else None
        )

        # Create model
        d_features = len(self.features)
        model = create_model(
            d_features=d_features,
            lstm_hidden=self.params.get("lstm_hidden", 64),
            hidden_units=self.params.get("hidden_units", [256, 128]),
            dropout=self.params.get("dropout", 0.2),
            device=self.device,
            model_type=self.params.get("model_type", "baseline"),
        )

        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=self.params.get("learning_rate", 1e-3)
        )

        # Training loop
        best_auc = 0
        patience_counter = 0

        for epoch in range(self.num_boost_round):
            # Training
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []

            for xs, seqs, seq_lens, ys in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
                xs, seqs, seq_lens, ys = (
                    xs.to(self.device),
                    seqs.to(self.device),
                    seq_lens.to(self.device),
                    ys.to(self.device),
                )

                optimizer.zero_grad()
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
                train_targets.extend(ys.cpu().detach().numpy())

            # Validation
            if valid_loader is not None:
                model.eval()
                valid_preds = []
                valid_targets = []

                with torch.no_grad():
                    for xs, seqs, seq_lens, ys in valid_loader:
                        xs, seqs, seq_lens, ys = (
                            xs.to(self.device),
                            seqs.to(self.device),
                            seq_lens.to(self.device),
                            ys.to(self.device),
                        )

                        logits = model(xs, seqs, seq_lens)
                        valid_preds.extend(torch.sigmoid(logits).cpu().detach().numpy())
                        valid_targets.extend(ys.cpu().detach().numpy())

                valid_auc = roc_auc_score(valid_targets, valid_preds)

                if self.verbose_eval > 0 and epoch % self.verbose_eval == 0:
                    print(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, Valid AUC: {valid_auc:.4f}"
                    )

                # Early stopping
                if valid_auc > best_auc:
                    best_auc = valid_auc
                    patience_counter = 0
                    # Save best model
                    torch.save(
                        model.state_dict(),
                        Path(self.model_path) / f"{self.results}.pth",
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_rounds:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                if self.verbose_eval > 0 and epoch % self.verbose_eval == 0:
                    print(
                        f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}"
                    )

        return model

    def _predict(
        self: Self, model: TabularSeqModel, X: pd.DataFrame | np.ndarray
    ) -> np.ndarray:
        """Make predictions using the trained model"""
        model.eval()
        test_loader = self._create_data_loader(X, None, is_training=False)
        predictions = []

        with torch.no_grad():
            for xs, seqs, seq_lens, _ in test_loader:
                xs, seqs, seq_lens = (
                    xs.to(self.device),
                    seqs.to(self.device),
                    seq_lens.to(self.device),
                )

                logits = model(xs, seqs, seq_lens)
                predictions.extend(torch.sigmoid(logits).cpu().detach().numpy())

        return np.array(predictions)

    def _create_data_loader(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None,
        is_training: bool,
    ) -> DataLoader:
        """Create data loader for training or inference"""
        # This is a simplified version - you may need to adapt based on your data structure
        if isinstance(X, pd.DataFrame):
            X = X[self.features].values

        # Create dummy sequence data for now - you'll need to implement proper sequence handling
        seq_data = np.zeros((X.shape[0], 10))  # Dummy sequence data
        seq_lengths = np.ones(X.shape[0]) * 10  # Dummy sequence lengths

        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            dataset = ClickDataset(X, seq_data, seq_lengths, y)
            return DataLoader(
                dataset,
                batch_size=4096,
                shuffle=is_training,
                collate_fn=collate_fn_train,
            )
        else:
            # For inference
            dataset = ClickDataset(X, seq_data, seq_lengths, np.zeros(X.shape[0]))
            return DataLoader(
                dataset, batch_size=4096, shuffle=False, collate_fn=collate_fn_train
            )

    def load_model(self: Self) -> TabularSeqModel:
        """Load the trained model"""
        d_features = len(self.features)
        model = create_model(
            d_features=d_features,
            lstm_hidden=self.params.get("lstm_hidden", 64),
            hidden_units=self.params.get("hidden_units", [256, 128]),
            dropout=self.params.get("dropout", 0.2),
            device=self.device,
            model_type=self.params.get("model_type", "baseline"),
        )
        model.load_state_dict(
            torch.load(
                Path(self.model_path) / f"{self.results}.pth", map_location=self.device
            )
        )
        return model
