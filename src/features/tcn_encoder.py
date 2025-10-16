import gc
import os
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from layers.tcn import create_history_tcn_encoder
from utils.config import load_yaml


class LSTMDecoder(nn.Module):
    """
    LSTM-based decoder for sequence reconstruction
    """

    def __init__(self, hidden_dim: int, seq_len: int = 30, lstm_hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm_hidden_dim = lstm_hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )

        # Output projection
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for layer in self.output_layer:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: Encoded representation (batch_size, hidden_dim)
        Returns:
            reconstructed: Reconstructed sequence (batch_size, seq_len)
        """
        # Repeat encoded representation for each timestep
        # (batch_size, hidden_dim) -> (batch_size, seq_len, hidden_dim)
        repeated_encoded = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Apply LSTM
        lstm_out, _ = self.lstm(
            repeated_encoded
        )  # (batch_size, seq_len, lstm_hidden_dim)

        # Apply output projection
        reconstructed = self.output_layer(lstm_out).squeeze(-1)  # (batch_size, seq_len)

        return reconstructed


class AttentionLSTMDecoder(nn.Module):
    """
    Improved Attention-based LSTM decoder with positional embedding and better normalization
    """

    def __init__(self, hidden_dim: int, seq_len: int = 30, lstm_hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm_hidden_dim = lstm_hidden_dim

        # Positional embedding for sequence position information
        self.pos_embedding = nn.Embedding(seq_len, hidden_dim)

        # Input projection layer
        self.input_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Enhanced LSTM layer with better configuration
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=3,  # Increased layers for better representation
            batch_first=True,
            dropout=0.2,
            bidirectional=True,  # Bidirectional for better context
        )

        # Adjust hidden dim for bidirectional LSTM
        lstm_output_dim = lstm_hidden_dim * 2  # Bidirectional

        # Multi-scale attention mechanism
        self.attention_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=lstm_output_dim,
                    num_heads=8,
                    dropout=0.2,
                    batch_first=True,
                ),
                nn.MultiheadAttention(
                    embed_dim=lstm_output_dim,
                    num_heads=4,
                    dropout=0.2,
                    batch_first=True,
                ),
            ]
        )

        # Attention fusion layer
        self.attention_fusion = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, lstm_output_dim),
            nn.LayerNorm(lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Enhanced output projection with residual connection
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Residual connection for output
        self.residual_projection = nn.Linear(lstm_output_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with better initialization"""
        # Initialize positional embedding
        nn.init.normal_(self.pos_embedding.weight, mean=0, std=0.1)

        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize attention layers
        for attention_layer in self.attention_layers:
            for name, param in attention_layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)

        # Initialize other layers
        modules_to_init = [
            self.input_projection,
            self.attention_fusion,
            self.output_layer,
        ]
        modules_to_init.append(self.residual_projection)  # Add single layer separately

        for module in modules_to_init:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: Encoded representation (batch_size, hidden_dim)
        Returns:
            reconstructed: Reconstructed sequence (batch_size, seq_len)
        """
        batch_size = encoded.size(0)

        # Create positional indices
        pos_indices = (
            torch.arange(self.seq_len, device=encoded.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        # Get positional embeddings
        pos_emb = self.pos_embedding(pos_indices)  # (batch_size, seq_len, hidden_dim)

        # Repeat encoded representation and add positional embedding
        repeated_encoded = encoded.unsqueeze(1).repeat(1, self.seq_len, 1)
        input_with_pos = repeated_encoded + pos_emb  # Add positional information

        # Apply input projection
        projected_input = self.input_projection(input_with_pos)

        # Apply bidirectional LSTM
        lstm_out, _ = self.lstm(
            projected_input
        )  # (batch_size, seq_len, lstm_hidden_dim * 2)

        # Apply multi-scale attention
        attention_outputs = []
        for attention_layer in self.attention_layers:
            attended_out, _ = attention_layer(lstm_out, lstm_out, lstm_out)
            attention_outputs.append(attended_out)

        # Fuse attention outputs
        fused_attention = torch.cat(attention_outputs, dim=-1)
        fused_output = self.attention_fusion(fused_attention)

        # Apply output projection with residual connection
        main_output = self.output_layer(fused_output).squeeze(-1)
        residual_output = self.residual_projection(fused_output).squeeze(-1)

        # Combine main output with residual
        reconstructed = main_output + 0.1 * residual_output  # Weighted residual

        return reconstructed


class TCNDataProcessor:
    """
    Data processor for TCN encoder training and inference
    """

    def __init__(self, config_path: str = "config/data/dataset.yaml"):
        self.config = load_yaml(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # History B columns
        self.history_b_cols = [f"history_b_{i}" for i in range(1, 31)]

    def load_data(self, data_path: str) -> pl.DataFrame:
        """Load data from parquet file"""
        return pl.read_parquet(data_path)

    def extract_history_sequences(self, df: pl.DataFrame) -> np.ndarray:
        """Extract history_b_1~30 sequences from dataframe"""
        history_data = df.select(self.history_b_cols).to_numpy()

        # Handle missing values by forward filling
        history_data = np.nan_to_num(history_data, nan=0.0)

        return history_data.astype(np.float32)

    def random_split(
        self,
        df: pl.DataFrame,
        val_split_ratio: float = 0.2,
        stratify_by_target: bool = True,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Improved random split with better data distribution

        Args:
            df: Input dataframe
            val_split_ratio: Ratio for validation split
            stratify_by_target: Whether to stratify by target variable (clicked)

        Returns:
            train_df, val_df: Split dataframes
        """
        import random

        random.seed(42)  # For reproducibility

        print("Using improved random split...")

        # Check if we can stratify by target variable
        if stratify_by_target and "clicked" in df.columns:
            print("Using stratified random split by target variable...")

            # Get unique target values and their counts
            target_counts = df["clicked"].value_counts().sort("clicked")

            train_indices = []
            val_indices = []

            for row in target_counts.iter_rows(named=True):
                target_value = row["clicked"]
                count = row["count"]

                # Get indices for this target value
                target_indices = (
                    df.filter(pl.col("clicked") == target_value)
                    .with_row_index("row_idx")
                    .select("row_idx")
                    .to_series()
                    .to_list()
                )

                # Shuffle indices for this target value
                random.shuffle(target_indices)

                # Split indices for this target value
                val_count = int(count * val_split_ratio)
                val_indices.extend(target_indices[:val_count])
                train_indices.extend(target_indices[val_count:])

            # Shuffle final indices
            random.shuffle(train_indices)
            random.shuffle(val_indices)

            # Create dataframes
            train_df = df[train_indices]
            val_df = df[val_indices]

            # Print statistics
            train_pos_ratio = train_df["clicked"].mean() if len(train_df) > 0 else 0
            val_pos_ratio = val_df["clicked"].mean() if len(val_df) > 0 else 0

            print("Stratified split:")
            print(
                f"  Train samples: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)"
            )
            print(
                f"  Validation samples: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)"
            )
            print(f"  Train positive ratio: {train_pos_ratio:.3f}")
            print(f"  Validation positive ratio: {val_pos_ratio:.3f}")

            return train_df, val_df

        # Fallback to simple random split
        print("Using simple random split...")

        # Create random indices
        total_size = len(df)
        indices = list(range(total_size))
        random.shuffle(indices)

        train_size = int((1 - val_split_ratio) * total_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_df = df[train_indices]
        val_df = df[val_indices]

        print("Random split:")
        print(
            f"  Train samples: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)"
        )
        print(
            f"  Validation samples: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)"
        )

        return train_df, val_df

    def stratify_split_by_column(
        self,
        df: pl.DataFrame,
        val_split_ratio: float = 0.2,
        stratify_column: Optional[str] = None,
        stratify_by_target: bool = True,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Stratify split data by specified column to ensure balanced distribution

        Args:
            df: Input dataframe
            val_split_ratio: Ratio for validation split
            stratify_column: Column to use for stratification (if None, uses improved random split)
            stratify_by_target: Whether to stratify by target variable when using random split

        Returns:
            train_df, val_df: Split dataframes
        """
        if stratify_column is None:
            return self.random_split(df, val_split_ratio, stratify_by_target)

        if stratify_column not in df.columns:
            print(
                f"Warning: {stratify_column} not found in data. Using improved random split instead."
            )
            return self.random_split(df, val_split_ratio, stratify_by_target)

        print(f"Performing stratify split by {stratify_column}...")

        # Get unique values in stratify column
        unique_values = df[stratify_column].unique().to_list()
        print(f"Found {len(unique_values)} unique {stratify_column}s")

        # Calculate how many unique values to put in validation
        n_val_values = int(len(unique_values) * val_split_ratio)

        # Randomly select values for validation
        import random

        random.seed(42)  # For reproducibility
        val_values = random.sample(unique_values, n_val_values)

        # Split data based on selected values
        val_df = df.filter(pl.col(stratify_column).is_in(val_values))
        train_df = df.filter(~pl.col(stratify_column).is_in(val_values))

        print(f"Train samples: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)")
        print(f"Validation samples: {len(val_df)} ({len(val_df) / len(df) * 100:.1f}%)")
        print(f"Train {stratify_column}s: {len(unique_values) - n_val_values}")
        print(f"Validation {stratify_column}s: {n_val_values}")

        return train_df, val_df

    def create_data_loader(
        self,
        history_sequences: np.ndarray,
        targets: Optional[np.ndarray] = None,
        batch_size: int = 1024,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create PyTorch DataLoader"""

        if targets is not None:
            dataset = TensorDataset(
                torch.from_numpy(history_sequences), torch.from_numpy(targets)
            )
        else:
            dataset = TensorDataset(torch.from_numpy(history_sequences))

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True if self.device.type == "cuda" else False,
        )


class TCNTrainer:
    """
    Trainer for TCN encoder with LSTM decoder
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        device: str = "cuda",
        decoder_type: str = "lstm",  # "lstm", "attention_lstm", "simple"
    ):
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.decoder_type = decoder_type

        # Initialize TCN encoder
        self.model = create_history_tcn_encoder(
            seq_len=30, hidden_dim=hidden_dim, pooling_method="attention"
        ).to(self.device)

        # Initialize decoder
        if decoder_type == "lstm":
            self.decoder = LSTMDecoder(
                hidden_dim=hidden_dim, seq_len=30, lstm_hidden_dim=64
            ).to(self.device)
        elif decoder_type == "attention_lstm":
            self.decoder = AttentionLSTMDecoder(
                hidden_dim=hidden_dim, seq_len=30, lstm_hidden_dim=64
            ).to(self.device)
        else:  # simple
            self.decoder = None

        # Initialize optimizer
        if self.decoder is not None:
            self.optimizer = optim.Adam(
                list(self.model.parameters()) + list(self.decoder.parameters()),
                lr=learning_rate,
            )
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

    def train_reconstruction(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train TCN encoder using reconstruction loss (self-supervised)
        """
        self.model.train()

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            train_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                history_seq = batch[0].to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                encoded = self.model(history_seq)

                # Reconstruction loss: try to reconstruct original sequence
                # Use encoded representation to predict the sequence
                reconstructed = self._reconstruct_sequence(encoded, history_seq.size(0))

                loss = self.criterion(reconstructed, history_seq)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            # Validation
            val_loss = self._validate_reconstruction(val_loader)

            avg_train_loss = train_loss / train_batches

            print(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model with decoder type
                model_path = f"models/tcn_encoder_best_{self.decoder_type}.pt"
                if self.decoder is not None:
                    # Save both encoder and decoder
                    torch.save(
                        {
                            "encoder_state_dict": self.model.state_dict(),
                            "decoder_state_dict": self.decoder.state_dict(),
                            "decoder_type": self.decoder_type,
                        },
                        model_path,
                    )
                else:
                    # Save only encoder
                    torch.save(self.model.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_model_path = f"models/tcn_encoder_best_{self.decoder_type}.pt"
        checkpoint = torch.load(best_model_path)
        if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["encoder_state_dict"])
            if self.decoder is not None and "decoder_state_dict" in checkpoint:
                self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def train_supervised(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Train TCN encoder using supervised loss (CTR prediction)
        """
        self.model.train()

        # Add classification head
        classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)

        classifier_optimizer = optim.Adam(
            classifier.parameters(), lr=self.learning_rate
        )
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            train_loss = 0.0
            train_batches = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}"):
                history_seq, targets = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                )

                self.optimizer.zero_grad()
                classifier_optimizer.zero_grad()

                # Forward pass
                encoded = self.model(history_seq)
                predictions = classifier(encoded).squeeze()

                loss = criterion(predictions, targets.float())

                # Backward pass
                loss.backward()
                self.optimizer.step()
                classifier_optimizer.step()

                train_loss += loss.item()
                train_batches += 1

            # Validation
            val_loss = self._validate_supervised(val_loader, classifier, criterion)

            avg_train_loss = train_loss / train_batches

            print(
                f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model with decoder type
                model_path = f"models/tcn_encoder_best_{self.decoder_type}.pt"
                if self.decoder is not None:
                    # Save both encoder and decoder
                    torch.save(
                        {
                            "encoder_state_dict": self.model.state_dict(),
                            "decoder_state_dict": self.decoder.state_dict(),
                            "decoder_type": self.decoder_type,
                        },
                        model_path,
                    )
                else:
                    # Save only encoder
                    torch.save(self.model.state_dict(), model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        best_model_path = f"models/tcn_encoder_best_{self.decoder_type}.pt"
        checkpoint = torch.load(best_model_path)
        if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["encoder_state_dict"])
            if self.decoder is not None and "decoder_state_dict" in checkpoint:
                self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

    def _reconstruct_sequence(
        self, encoded: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """
        Advanced reconstruction method using LSTM decoder
        """
        if self.decoder is not None:
            # Use LSTM decoder for reconstruction
            reconstructed = self.decoder(encoded)
        else:
            # Fallback to simple reconstruction
            seq_len = 30
            reconstructed = encoded.unsqueeze(1).repeat(1, seq_len, 1)
            reconstructed = reconstructed.mean(dim=-1)  # Average over hidden dim

        return reconstructed

    def _validate_reconstruction(self, val_loader: DataLoader) -> float:
        """Validate reconstruction performance"""
        self.model.eval()
        if self.decoder is not None:
            self.decoder.eval()

        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                history_seq = batch[0].to(self.device)

                encoded = self.model(history_seq)
                reconstructed = self._reconstruct_sequence(encoded, history_seq.size(0))

                loss = self.criterion(reconstructed, history_seq)
                val_loss += loss.item()
                val_batches += 1

        self.model.train()
        if self.decoder is not None:
            self.decoder.train()
        return val_loss / val_batches

    def _validate_supervised(
        self, val_loader: DataLoader, classifier: nn.Module, criterion: nn.Module
    ) -> float:
        """Validate supervised performance"""
        self.model.eval()
        classifier.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                history_seq, targets = (
                    batch[0].to(self.device),
                    batch[1].to(self.device),
                )

                encoded = self.model(history_seq)
                predictions = classifier(encoded).squeeze()

                loss = criterion(predictions, targets.float())
                val_loss += loss.item()
                val_batches += 1

        self.model.train()
        classifier.train()
        return val_loss / val_batches


class TCNEncoder:
    """
    Main TCN encoder class for training and inference
    """

    def __init__(self, config_path: str = "config/data/dataset.yaml"):
        self.processor = TCNDataProcessor(config_path)
        self.trainer = None
        self.model = None

    def train(
        self,
        train_data_path: str,
        val_data_path: str = None,
        val_split_ratio: float = 0.2,
        stratify_column: Optional[str] = None,
        stratify_by_target: bool = True,
        training_mode: str = "reconstruction",
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        decoder_type: str = "attention_lstm",
    ):
        """
        Train TCN encoder

        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data (if None, will split from train_data)
            val_split_ratio: Ratio for validation split when splitting from train data
            stratify_column: Column to use for stratification (if None, uses improved random split)
            stratify_by_target: Whether to stratify by target variable when using random split
            training_mode: "reconstruction" or "supervised"
            hidden_dim: Output embedding dimension
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"Training TCN encoder in {training_mode} mode...")

        # Load data
        train_df = self.processor.load_data(train_data_path)

        # Handle validation data
        if val_data_path is None:
            print(
                f"Splitting training data into train/validation (ratio: {1 - val_split_ratio:.1f}/{val_split_ratio:.1f})..."
            )
            # Use stratify split if column is specified
            train_df, val_df = self.processor.stratify_split_by_column(
                train_df, val_split_ratio, stratify_column, stratify_by_target
            )
            print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")
        else:
            print(f"Loading validation data from: {val_data_path}")
            val_df = self.processor.load_data(val_data_path)
            print(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

        # Extract history sequences
        train_history = self.processor.extract_history_sequences(train_df)
        val_history = self.processor.extract_history_sequences(val_df)

        # Create data loaders
        if training_mode == "supervised":
            train_targets = train_df.select("clicked").to_numpy().flatten()
            val_targets = val_df.select("clicked").to_numpy().flatten()

            train_loader = self.processor.create_data_loader(
                train_history, train_targets, batch_size, shuffle=True
            )
            val_loader = self.processor.create_data_loader(
                val_history, val_targets, batch_size, shuffle=False
            )
        else:
            train_loader = self.processor.create_data_loader(
                train_history, None, batch_size, shuffle=True
            )
            val_loader = self.processor.create_data_loader(
                val_history, None, batch_size, shuffle=False
            )

        # Initialize trainer
        self.trainer = TCNTrainer(
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            epochs=epochs,
            device="cuda",
            decoder_type=decoder_type,
        )

        # Train model
        if training_mode == "supervised":
            self.trainer.train_supervised(train_loader, val_loader)
        else:
            self.trainer.train_reconstruction(train_loader, val_loader)

        # Save final model with decoder type
        self.model = self.trainer.model
        final_model_path = f"models/tcn_encoder_final_{decoder_type}.pt"
        if self.trainer.decoder is not None:
            # Save both encoder and decoder
            torch.save(
                {
                    "encoder_state_dict": self.model.state_dict(),
                    "decoder_state_dict": self.trainer.decoder.state_dict(),
                    "decoder_type": decoder_type,
                },
                final_model_path,
            )
        else:
            # Save only encoder
            torch.save(self.model.state_dict(), final_model_path)

        print("TCN encoder training completed!")

    def train_kfold(
        self,
        train_data_path: str,
        test_data_path: str,
        output_dir: str,
        n_splits: int = 5,
        random_state: int = 42,
        training_mode: str = "reconstruction",
        hidden_dim: int = 32,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 1024,
        decoder_type: str = "attention_lstm",
    ):
        """
        Train TCN encoder with K-Fold CV and encode data for each fold

        This ensures no CV leakage by training separate models for each fold
        and encoding train/test data accordingly.

        Args:
            train_data_path: Path to training data
            test_data_path: Path to test data
            output_dir: Directory to save encoded data
            n_splits: Number of folds for K-Fold CV
            random_state: Random state for reproducibility (must match DCN+MHA)
            training_mode: "reconstruction" or "supervised"
            hidden_dim: Output embedding dimension
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            decoder_type: Type of decoder to use
        """
        print("=" * 80)
        print(f"Training TCN encoder with {n_splits}-Fold Cross-Validation")
        print("=" * 80)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load full training data
        print(f"Loading training data from {train_data_path}...")
        full_train_df = self.processor.load_data(train_data_path)
        print(f"Total training samples: {len(full_train_df):,}")

        # Load test data (will be encoded with each fold's model)
        print(f"Loading test data from {test_data_path}...")
        test_df = self.processor.load_data(test_data_path)
        print(f"Total test samples: {len(test_df):,}")

        # Extract history sequences for full data
        full_history = self.processor.extract_history_sequences(full_train_df)
        test_history = self.processor.extract_history_sequences(test_df)

        # Get target for stratification
        if "clicked" not in full_train_df.columns:
            raise ValueError("Target column 'clicked' not found in training data")

        y = full_train_df.select("clicked").to_numpy().flatten()

        # Initialize K-Fold splitter (same as DCN+MHA training)
        kfold = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

        print("\nK-Fold settings:")
        print(f"  - n_splits: {n_splits}")
        print(f"  - random_state: {random_state}")
        print("  - shuffle: True")
        print("  - stratified: True (by 'clicked' target)")

        # Train and encode for each fold
        for fold_idx, (train_indices, val_indices) in enumerate(
            kfold.split(full_history, y), 1
        ):
            print("\n" + "=" * 80)
            print(f"FOLD {fold_idx}/{n_splits}")
            print("=" * 80)

            # Split data for this fold
            train_fold_history = full_history[train_indices]
            val_fold_history = full_history[val_indices]

            train_fold_df = full_train_df[train_indices]
            val_fold_df = full_train_df[val_indices]

            print(f"Train samples: {len(train_indices):,}")
            print(f"Val samples: {len(val_indices):,}")

            # Check target distribution
            train_pos_ratio = train_fold_df.select("clicked").mean().item()
            val_pos_ratio = val_fold_df.select("clicked").mean().item()
            print(f"Train positive ratio: {train_pos_ratio:.4f}")
            print(f"Val positive ratio: {val_pos_ratio:.4f}")

            # Create data loaders
            if training_mode == "supervised":
                train_targets = train_fold_df.select("clicked").to_numpy().flatten()
                val_targets = val_fold_df.select("clicked").to_numpy().flatten()

                train_loader = self.processor.create_data_loader(
                    train_fold_history, train_targets, batch_size, shuffle=True
                )
                val_loader = self.processor.create_data_loader(
                    val_fold_history, val_targets, batch_size, shuffle=False
                )
            else:
                train_loader = self.processor.create_data_loader(
                    train_fold_history, None, batch_size, shuffle=True
                )
                val_loader = self.processor.create_data_loader(
                    val_fold_history, None, batch_size, shuffle=False
                )

            # Initialize trainer for this fold
            print(f"\nInitializing TCN trainer for fold {fold_idx}...")
            self.trainer = TCNTrainer(
                hidden_dim=hidden_dim,
                learning_rate=learning_rate,
                epochs=epochs,
                device="cuda",
                decoder_type=decoder_type,
            )

            # Train model for this fold
            print(f"\nTraining TCN encoder for fold {fold_idx}...")
            if training_mode == "supervised":
                self.trainer.train_supervised(train_loader, val_loader)
            else:
                self.trainer.train_reconstruction(train_loader, val_loader)

            # Save fold model
            self.model = self.trainer.model
            fold_model_path = f"models/tcn_encoder_fold{fold_idx}_{decoder_type}.pt"
            if self.trainer.decoder is not None:
                torch.save(
                    {
                        "encoder_state_dict": self.model.state_dict(),
                        "decoder_state_dict": self.trainer.decoder.state_dict(),
                        "decoder_type": decoder_type,
                        "fold": fold_idx,
                    },
                    fold_model_path,
                )
            else:
                torch.save(self.model.state_dict(), fold_model_path)
            print(f"Saved fold {fold_idx} model to {fold_model_path}")

            # Encode training data for this fold
            print(f"\nEncoding training data for fold {fold_idx}...")
            train_encoded = self._encode_batch(full_history, batch_size)

            # Create TCN embedding columns
            tcn_columns = [f"tcn_emb_{i + 1}" for i in range(train_encoded.shape[1])]
            tcn_df = pl.DataFrame(train_encoded, schema=tcn_columns)

            # Combine with original data
            train_with_tcn = pl.concat([full_train_df, tcn_df], how="horizontal")

            # Save training data with TCN embeddings
            train_output_path = os.path.join(
                output_dir, f"train_with_tcn_fold{fold_idx}.parquet"
            )
            train_with_tcn.write_parquet(train_output_path)
            print(f"Saved: {train_output_path}")
            print(f"  - Shape: {train_with_tcn.shape}")
            print(f"  - TCN features: {len(tcn_columns)}")

            # Encode test data with this fold's model
            print(f"\nEncoding test data for fold {fold_idx}...")
            test_encoded = self._encode_batch(test_history, batch_size)

            # Create TCN embedding columns for test
            test_tcn_df = pl.DataFrame(test_encoded, schema=tcn_columns)

            # Combine with original test data
            test_with_tcn = pl.concat([test_df, test_tcn_df], how="horizontal")

            # Save test data with TCN embeddings
            test_output_path = os.path.join(
                output_dir, f"test_with_tcn_fold{fold_idx}.parquet"
            )
            test_with_tcn.write_parquet(test_output_path)
            print(f"Saved: {test_output_path}")
            print(f"  - Shape: {test_with_tcn.shape}")
            print(f"  - TCN features: {len(tcn_columns)}")

            # Clean up memory
            del train_loader, val_loader, train_fold_history, val_fold_history
            del train_fold_df, val_fold_df, train_encoded, test_encoded
            del tcn_df, test_tcn_df, train_with_tcn, test_with_tcn
            torch.cuda.empty_cache()
            gc.collect()

            print(f"\nFold {fold_idx} completed!")

        print("\n" + "=" * 80)
        print("K-Fold TCN training and encoding completed!")
        print(f"Generated {n_splits} train and {n_splits} test files")
        print("=" * 80)

    def _encode_batch(
        self, history_sequences: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """
        Encode history sequences in batches

        Args:
            history_sequences: History sequences to encode
            batch_size: Batch size for encoding

        Returns:
            Encoded features as numpy array
        """
        self.model.eval()
        encoded_features = []

        # Create data loader
        data_loader = self.processor.create_data_loader(
            history_sequences, None, batch_size=batch_size, shuffle=False
        )

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Encoding", leave=False):
                history_seq = batch[0].to(self.processor.device)
                encoded = self.model(history_seq)
                encoded_features.append(encoded.cpu().numpy())

        # Concatenate all encoded features
        return np.concatenate(encoded_features, axis=0)

    def encode_data(self, data_path: str, output_path: str, model_path: str = None):
        """
        Encode data using trained TCN encoder

        Args:
            data_path: Path to input data
            output_path: Path to save encoded data
        """
        if self.model is None:
            # Load trained model - try different decoder types
            self.model = create_history_tcn_encoder(
                seq_len=30, hidden_dim=32, pooling_method="attention"
            )

            # Try to load model with decoder type in filename
            if model_path is not None and os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.processor.device)
                if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["encoder_state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
                print(f"Loaded model: {model_path}")

            self.model.to(self.processor.device)

        print(f"Encoding data from {data_path}...")

        # Load data
        df = self.processor.load_data(data_path)

        # Extract history sequences
        history_sequences = self.processor.extract_history_sequences(df)

        # Create data loader
        data_loader = self.processor.create_data_loader(
            history_sequences, None, batch_size=1024, shuffle=False
        )

        # Encode data
        self.model.eval()
        encoded_features = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Encoding"):
                history_seq = batch[0].to(self.processor.device)
                encoded = self.model(history_seq)
                encoded_features.append(encoded.cpu().numpy())

        # Concatenate all encoded features
        encoded_features = np.concatenate(encoded_features, axis=0)

        # Create new dataframe with TCN embeddings
        tcn_columns = [f"tcn_emb_{i + 1}" for i in range(encoded_features.shape[1])]
        tcn_df = pl.DataFrame(encoded_features, schema=tcn_columns)

        # Combine with original data
        result_df = pl.concat([df, tcn_df], how="horizontal")

        # Save result
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_df.write_parquet(output_path)

        print(f"Encoded data saved to {output_path}")
        print(f"Added {len(tcn_columns)} TCN embedding features")
